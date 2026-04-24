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
bpy.ops.import_scene.fbx(filepath=fbx_in)
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
g.save(path)
PYEOF
