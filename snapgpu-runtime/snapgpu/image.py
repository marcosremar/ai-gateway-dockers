"""Container image builder (Modal-style builder pattern)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImageStep:
    """A single step in the image build process."""
    type: str  # "from", "run", "pip", "apt", "copy", "env", "workdir"
    args: list[str] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)

    def to_dockerfile_line(self) -> str:
        if self.type == "from":
            return f"FROM {self.args[0]}"
        elif self.type == "run":
            return f"RUN {self.args[0]}"
        elif self.type == "pip":
            pkgs = " ".join(f'"{p}"' for p in self.args)
            extra = ""
            if self.kwargs.get("find_links"):
                extra += f" --find-links {self.kwargs['find_links']}"
            if self.kwargs.get("index_url"):
                extra += f" --index-url {self.kwargs['index_url']}"
            if self.kwargs.get("extra_index_url"):
                extra += f" --extra-index-url {self.kwargs['extra_index_url']}"
            return f"RUN pip install --no-cache-dir{extra} {pkgs}"
        elif self.type == "apt":
            pkgs = " ".join(self.args)
            return f"RUN apt-get update -qq && apt-get install -y --no-install-recommends {pkgs} && rm -rf /var/lib/apt/lists/*"
        elif self.type == "copy":
            return f"COPY {self.args[0]} {self.args[1]}"
        elif self.type == "env":
            return f"ENV {self.args[0]}={self.args[1]}"
        elif self.type == "workdir":
            return f"WORKDIR {self.args[0]}"
        else:
            raise ValueError(f"Unknown step type: {self.type}")


class Image:
    """Declarative container image builder.

    Usage:
        image = (
            snapgpu.Image.from_registry("nvidia/cuda:12.1.0-runtime-ubuntu22.04")
            .apt_install("ffmpeg", "libsndfile1")
            .pip_install("torch", "transformers")
            .run_commands("python -c 'import torch; print(torch.cuda.is_available())'")
        )
    """

    def __init__(self, steps: Optional[list[ImageStep]] = None):
        self._steps: list[ImageStep] = steps or []

    def _clone_with(self, step: ImageStep) -> Image:
        """Return a new Image with the step appended (immutable builder)."""
        return Image(steps=self._steps + [step])

    # ── Factory methods ──────────────────────────────────────────────────

    @classmethod
    def from_registry(cls, base: str) -> Image:
        """Start from a Docker registry image."""
        return cls(steps=[ImageStep(type="from", args=[base])])

    @classmethod
    def debian_slim(cls, python_version: str = "3.11") -> Image:
        """Start from a slim Debian image with Python."""
        return cls(steps=[ImageStep(type="from", args=[f"python:{python_version}-slim"])])

    @classmethod
    def from_dockerfile(cls, path: str) -> Image:
        """Start from an existing Dockerfile."""
        img = cls()
        img._dockerfile_path = path
        return img

    # ── Builder methods ──────────────────────────────────────────────────

    def pip_install(self, *packages: str, find_links: str = "",
                    index_url: str = "", extra_index_url: str = "") -> Image:
        """Install Python packages via pip."""
        kwargs = {}
        if find_links:
            kwargs["find_links"] = find_links
        if index_url:
            kwargs["index_url"] = index_url
        if extra_index_url:
            kwargs["extra_index_url"] = extra_index_url
        return self._clone_with(ImageStep(type="pip", args=list(packages), kwargs=kwargs))

    def apt_install(self, *packages: str) -> Image:
        """Install system packages via apt-get."""
        return self._clone_with(ImageStep(type="apt", args=list(packages)))

    def run_commands(self, *commands: str) -> Image:
        """Run shell commands during build."""
        img = self
        for cmd in commands:
            img = img._clone_with(ImageStep(type="run", args=[cmd]))
        return img

    def copy_local_file(self, src: str, dst: str) -> Image:
        """Copy a local file into the image."""
        return self._clone_with(ImageStep(type="copy", args=[src, dst]))

    def env(self, **kwargs: str) -> Image:
        """Set environment variables."""
        img = self
        for k, v in kwargs.items():
            img = img._clone_with(ImageStep(type="env", args=[k, v]))
        return img

    def workdir(self, path: str) -> Image:
        """Set working directory."""
        return self._clone_with(ImageStep(type="workdir", args=[path]))

    # ── Output ───────────────────────────────────────────────────────────

    def to_dockerfile(self) -> str:
        """Generate Dockerfile content from steps."""
        if hasattr(self, "_dockerfile_path"):
            with open(self._dockerfile_path) as f:
                return f.read()
        lines = [step.to_dockerfile_line() for step in self._steps]
        # Ensure python pip is available
        if any(s.type == "pip" for s in self._steps):
            base = self._steps[0].args[0] if self._steps else ""
            if "python" not in base.lower() and "pip" not in "\n".join(lines):
                lines.insert(1, "RUN apt-get update -qq && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*")
        return "\n".join(lines) + "\n"

    @property
    def base_image(self) -> str:
        """Return the base image name."""
        for step in self._steps:
            if step.type == "from":
                return step.args[0]
        return "python:3.11-slim"

    def __repr__(self) -> str:
        return f"Image(steps={len(self._steps)}, base={self.base_image!r})"
