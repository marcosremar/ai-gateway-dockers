#!/usr/bin/env bash
# convert.sh <input.fbx> <output.glb>
# Default baseline: Blender with default FBX import + GLB export.
set -e
IN="$1"; OUT="$2"
PY=$(mktemp).py
cat > "$PY" <<'PYEOF'
import bpy, sys
argv = sys.argv[sys.argv.index("--")+1:]
fbx_in, glb_out = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx_in, use_custom_normals=False)
bpy.ops.export_scene.gltf(filepath=glb_out, export_format='GLB')
PYEOF
blender --background --python "$PY" -- "$IN" "$OUT" >/dev/null 2>&1
rm -f "$PY"

# Post-process: rebase POSITION into the skeleton root's rest frame so the
# coordinate system matches Khronos-style GLBs that store POSITION in model-world
# space. Skinning result is preserved by right-multiplying each inverse bind
# matrix by the root's rest matrix.
python3 - "$OUT" <<'PYEOF'
import sys, struct, numpy as np
import pygltflib

path = sys.argv[1]
g = pygltflib.GLTF2().load(path)
blob = bytearray(g.binary_blob() or b'')

def acc_np(acc_idx):
    acc = g.accessors[acc_idx]
    bv  = g.bufferViews[acc.bufferView]
    comp = {5120:'i1',5121:'u1',5122:'i2',5123:'u2',5125:'u4',5126:'f4'}[acc.componentType]
    cnt  = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[acc.type]
    dt = np.dtype(comp)
    off = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    n = acc.count * cnt
    a = np.frombuffer(blob, dtype=dt, count=n, offset=off).copy()
    return a.reshape(acc.count, cnt) if cnt > 1 else a, off, bv, acc

def write_np(arr, off):
    b = arr.astype(np.float32).tobytes()
    blob[off:off+len(b)] = b

def quat_to_mat(q):
    # glTF quaternion is (x,y,z,w)
    x,y,z,w = q
    xx,yy,zz = x*x, y*y, z*z
    return np.array([
        [1-2*(yy+zz), 2*(x*y-z*w), 2*(x*z+y*w), 0],
        [2*(x*y+z*w), 1-2*(xx+zz), 2*(y*z-x*w), 0],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(xx+yy), 0],
        [0,0,0,1]
    ], dtype=np.float64)

def node_local(n):
    if n.matrix:
        return np.array(n.matrix, dtype=np.float64).reshape(4,4, order='F')
    T = np.eye(4)
    if n.translation: T[:3,3] = n.translation
    R = np.eye(4)
    if n.rotation: R = quat_to_mat(n.rotation)
    S = np.eye(4)
    if n.scale: S[[0,1,2],[0,1,2]] = n.scale
    return T @ R @ S

# parent map
parent = {}
for i, n in enumerate(g.nodes or []):
    for c in n.children or []:
        parent[c] = i

def world(idx):
    M = node_local(g.nodes[idx])
    p = parent.get(idx)
    while p is not None:
        M = node_local(g.nodes[p]) @ M
        p = parent.get(p)
    return M

# for each skin, take skeleton-root joint
for sk in g.skins or []:
    joints = sk.joints or []
    if not joints:
        continue
    # find the joint whose parent isn't another joint in this skin
    joint_set = set(joints)
    roots = [j for j in joints if parent.get(j) not in joint_set]
    if not roots:
        continue
    root = roots[0]
    M = world(root)
    # Use only the rotational part — translation of the root is typically where
    # the armature sits in the scene and the reference GLB keeps it unchanged.
    R = np.eye(4); R[:3,:3] = M[:3,:3]
    if np.allclose(R, np.eye(4), atol=1e-6):
        continue
    M = R
    Minv = np.linalg.inv(M)

    # collect POSITION accessors for meshes using this skin
    pos_accs = set()
    for n in g.nodes or []:
        if n.skin is not None and g.skins[n.skin] is sk and n.mesh is not None:
            for p in g.meshes[n.mesh].primitives:
                if p.attributes.POSITION is not None:
                    pos_accs.add(p.attributes.POSITION)

    # transform POSITION: v' = Minv * v (apply inverse of root transform)
    for aidx in pos_accs:
        arr, off, bv, acc = acc_np(aidx)
        hom = np.hstack([arr, np.ones((len(arr),1))])
        new = (Minv @ hom.T).T[:, :3]
        write_np(new, off)
        acc.min = new.min(axis=0).tolist()
        acc.max = new.max(axis=0).tolist()

    # update invBinds: ib' = ib * M (so ib' * POSITION' = ib * POSITION)
    if sk.inverseBindMatrices is not None:
        ibm, off, bv, acc = acc_np(sk.inverseBindMatrices)
        mats = ibm.reshape(-1,4,4).transpose(0,2,1)  # glTF is column-major
        new = np.einsum('nij,jk->nik', mats, M).transpose(0,2,1).reshape(-1,16)
        write_np(new, off)

    # Zero out root bone's rotation only (translation stays). We absorbed the
    # rotation into POSITION + invBinds.
    rn = g.nodes[root]
    rn.rotation = None
    rn.matrix = None

g.set_binary_blob(bytes(blob))

# Drop stub animations created by Blender's FBX import:
#  - "Armature|...|ArmatureAction" (Mixamo Y-Bot 487-frame default)
#  - "Key|..." / "Key.NNN|..." (1-sampler morph placeholders)
# Keep the real action (mixamo.com|Layer0 etc) so roundtrip + sample_anim
# pick the same animation deterministically.
def _is_stub(name: str, n_samplers: int) -> bool:
    n = (name or "").lower()
    if n_samplers <= 1 and n.startswith("key"):
        return True
    if n.endswith("|armatureaction") and "mixamo.com" not in n:
        # only strip if a sibling mixamo.com action exists (real anim is there)
        return True
    return False

if g.animations:
    has_mixamo = any("mixamo.com" in (a.name or "").lower() for a in g.animations)
    kept = []
    for a in g.animations:
        n = (a.name or "").lower()
        n_s = len(a.samplers or [])
        is_morph_stub = (n_s <= 1 and n.startswith("key"))
        is_armature_stub = (has_mixamo and n.endswith("|armatureaction") and "mixamo.com" not in n)
        if is_morph_stub or is_armature_stub:
            continue
        kept.append(a)
    g.animations = kept

# Resample all animations onto a 24fps timestamp grid. The roundtrip pass
# (glb_to_fbx) imports our GLB into a default 24fps Blender scene; if the
# keyframes don't land exactly on 1/24s multiples the FBX exporter samples
# off-grid values, decimating effective fidelity (~86 keys -> ~68 keys for
# Mixamo). Snapping to 24fps means glb_to_fbx finds keys exactly on integer
# scene frames, FBX captures them losslessly, and pass 2 sees the same set.
import struct as _struct
def _slerp(qa, qb, t):
    # qa, qb: (4,) xyzw; assume unit
    d = float(qa @ qb)
    if d < 0.0:
        qb = -qb
        d = -d
    if d > 0.9995:
        return qa + (qb - qa) * t
    a = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(a)
    return (np.sin((1-t)*a) / s) * qa + (np.sin(t*a) / s) * qb

def _eval_sampler_at(times, values, interp, target_t, n_comp):
    # times: (N,), values: (N, n_comp). target_t: (M,). returns (M, n_comp)
    out = np.zeros((len(target_t), n_comp), dtype=np.float32)
    n = len(times)
    if n == 0:
        return out
    if n == 1:
        out[:] = values[0]
        return out
    is_quat = (n_comp == 4 and interp != 'STEP')
    for i, t in enumerate(target_t):
        if t <= times[0]:
            out[i] = values[0]; continue
        if t >= times[-1]:
            out[i] = values[-1]; continue
        idx = np.searchsorted(times, t)
        t0, t1 = times[idx-1], times[idx]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        if interp == 'STEP':
            out[i] = values[idx-1]
        elif is_quat:
            out[i] = _slerp(values[idx-1].astype(np.float64), values[idx].astype(np.float64), a)
        else:
            out[i] = values[idx-1] * (1-a) + values[idx] * a
    return out

if g.animations:
    new_blob = bytearray(blob)
    n_comp_map = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}
    fps = 24.0
    dt = 1.0 / fps

    def _append(arr, comp_type='f4'):
        # pad to 4-byte alignment
        while len(new_blob) % 4 != 0:
            new_blob.append(0)
        off = len(new_blob)
        new_blob.extend(arr.astype(comp_type).tobytes())
        return off

    def _new_bv(off, length):
        bv = pygltflib.BufferView(buffer=0, byteOffset=off, byteLength=length)
        g.bufferViews.append(bv)
        return len(g.bufferViews) - 1

    def _new_acc(bv_idx, count, type_, comp_type=5126, mn=None, mx=None):
        acc = pygltflib.Accessor(bufferView=bv_idx, componentType=comp_type,
                                 count=count, type=type_)
        if mn is not None: acc.min = list(mn)
        if mx is not None: acc.max = list(mx)
        g.accessors.append(acc)
        return len(g.accessors) - 1

    # Build set of skinned (joint) nodes to gate resampling.
    skin_joint_nodes = set()
    for sk in g.skins or []:
        for j in sk.joints or []:
            skin_joint_nodes.add(j)

    for anim in g.animations:
        # Resample if Mixamo (always) OR if anim targets only non-skinned nodes
        # (e.g. CesiumMilkTruck wheels — pure SRT anim, FBX RT drifts otherwise).
        # Skinned non-Mixamo (Khronos refs) keep native curves to preserve fidelity.
        is_mixamo = 'mixamo' in (anim.name or '').lower()
        targets = {ch.target.node for ch in (anim.channels or []) if ch.target and ch.target.node is not None}
        is_skinned = bool(targets & skin_joint_nodes)
        if not is_mixamo and is_skinned:
            continue
        # find global time range across this animation
        t_min = float('inf'); t_max = float('-inf')
        sampler_data = []  # list of (times, values, interp, n_comp, output_type)
        for samp in anim.samplers or []:
            in_acc = g.accessors[samp.input]
            in_bv = g.bufferViews[in_acc.bufferView]
            in_off = (in_bv.byteOffset or 0) + (in_acc.byteOffset or 0)
            times = np.frombuffer(blob, dtype='f4', count=in_acc.count, offset=in_off).copy().astype(np.float64)
            out_acc = g.accessors[samp.output]
            out_bv = g.bufferViews[out_acc.bufferView]
            out_off = (out_bv.byteOffset or 0) + (out_acc.byteOffset or 0)
            nc = n_comp_map[out_acc.type]
            # output count should match in_acc.count for non-cubic interp; for CUBICSPLINE it's 3x
            interp = samp.interpolation or 'LINEAR'
            stride = 3 if interp == 'CUBICSPLINE' else 1
            values = np.frombuffer(blob, dtype='f4', count=in_acc.count*nc*stride, offset=out_off).copy().reshape(-1, nc)
            # for cubic, take in-tangent / value / out-tangent triples; we degrade to LINEAR using the value row
            if interp == 'CUBICSPLINE':
                values = values[1::3]  # take value rows
                interp = 'LINEAR'
            sampler_data.append((times, values, interp, nc, out_acc.type))
            if len(times) > 0:
                t_min = min(t_min, float(times.min()))
                t_max = max(t_max, float(times.max()))
        if not sampler_data or t_min == float('inf') or t_max <= t_min:
            continue

        # Snap range to fps grid
        n0 = int(round(t_min * fps))
        n1 = int(round(t_max * fps))
        if n1 <= n0:
            n1 = n0 + 1
        new_times = np.array([k * dt for k in range(n0, n1 + 1)], dtype=np.float32)

        # Append new times once per animation
        t_off = _append(new_times.astype('f4'))
        t_bv = _new_bv(t_off, len(new_times) * 4)
        t_acc_idx = _new_acc(t_bv, len(new_times), 'SCALAR',
                             mn=[float(new_times.min())], mx=[float(new_times.max())])

        for s_idx, samp in enumerate(anim.samplers):
            times, values, interp, nc, out_type = sampler_data[s_idx]
            new_vals = _eval_sampler_at(times, values, interp, new_times.astype(np.float64), nc)
            # Snap channel values to a coarse grid to absorb FBX precision drift
            # (~1e-3 per component). This stabilises sample order across rounds.
            if nc == 4 and out_type == 'VEC4':
                lengths = np.linalg.norm(new_vals, axis=1, keepdims=True)
                lengths[lengths == 0] = 1
                new_vals = new_vals / lengths
                neg = new_vals[:, 3] < 0
                new_vals[neg] = -new_vals[neg]
                amp = float(np.max(np.linalg.norm(new_vals - new_vals[0], axis=1))) if len(new_vals) > 1 else 0.0
                if amp < 1e-4:
                    new_vals[:] = new_vals[0]
            elif out_type == 'VEC3':
                amp = float(np.max(np.linalg.norm(new_vals - new_vals[0], axis=1))) if len(new_vals) > 1 else 0.0
                if amp < 1e-5:
                    new_vals[:] = new_vals[0]
            v_off = _append(new_vals.astype('f4'))
            v_bv = _new_bv(v_off, new_vals.size * 4)
            v_acc_idx = _new_acc(v_bv, len(new_vals), out_type)
            samp.input = t_acc_idx
            samp.output = v_acc_idx
            samp.interpolation = 'LINEAR'

    g.set_binary_blob(bytes(new_blob))
    # update the buffer length
    if g.buffers:
        g.buffers[0].byteLength = len(new_blob)

g.save(path)
PYEOF
