"""Convert Hugging Face safetensors models into Ollama-ready GGUF models.

Three public entry points used by the CLI:

- ``convert_hf_repo``: download safetensors, convert to GGUF, optionally
  quantize, then register with Ollama via ``ollama create``.
- ``import_gguf``: register an existing local GGUF file with Ollama.
- ``setup_llama_cpp``: clone llama.cpp into ``~/.anvil/llama.cpp`` for
  the convert/quantize binaries.

Heavy dependencies (``huggingface_hub``, ``torch``, ``transformers``) are
not anvil-wide requirements. They are pulled in only when the user runs
convert, via the ``[convert]`` extra:

    pip install -e '.[convert]'

llama.cpp itself is an external tool. We locate it via (in order):
1. ``LLAMA_CPP_PATH`` environment variable
2. ``~/.anvil/llama.cpp`` (created by ``setup_llama_cpp``)
3. ``PATH`` (for the ``llama-quantize`` binary only)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from anvil.utils.logging import get_logger

log = get_logger("llm.convert")

__all__ = [
    "ConvertResult",
    "DEFAULT_LLAMA_CPP_DIR",
    "convert_hf_repo",
    "import_gguf",
    "setup_llama_cpp",
    "find_convert_script",
    "find_quantize_binary",
    "preflight",
]

DEFAULT_LLAMA_CPP_DIR = Path.home() / ".anvil" / "llama.cpp"
LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"

# Rough safetensors->GGUF size multipliers. A 7B model at fp16 safetensors is
# ~14 GB; convert to f16 GGUF adds another ~14 GB; quantize to Q4 adds ~4 GB.
# So peak disk during convert is ~2x source + quantized output.
DISK_OVERHEAD_FACTOR = 2.2


@dataclass(slots=True)
class ConvertResult:
    """Outcome of a convert_hf_repo call."""

    ok: bool
    message: str
    gguf_path: Path | None = None
    ollama_name: str | None = None


def find_convert_script(llama_cpp_dir: Path | None = None) -> Path | None:
    """Locate llama.cpp's ``convert_hf_to_gguf.py``. Returns None if not found."""
    candidates: list[Path] = []
    env_path = os.environ.get("LLAMA_CPP_PATH")
    if env_path:
        candidates.append(Path(env_path))
    if llama_cpp_dir:
        candidates.append(llama_cpp_dir)
    candidates.append(DEFAULT_LLAMA_CPP_DIR)

    for root in candidates:
        for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
            p = root / name
            if p.is_file():
                return p
    return None


def find_quantize_binary(llama_cpp_dir: Path | None = None) -> Path | None:
    """Locate llama.cpp's ``llama-quantize`` binary. Returns None if not found."""
    candidates: list[Path] = []
    env_path = os.environ.get("LLAMA_CPP_PATH")
    if env_path:
        candidates.extend([Path(env_path) / "build" / "bin" / "llama-quantize", Path(env_path) / "llama-quantize"])
    if llama_cpp_dir:
        candidates.extend([llama_cpp_dir / "build" / "bin" / "llama-quantize", llama_cpp_dir / "llama-quantize"])
    candidates.extend([
        DEFAULT_LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
        DEFAULT_LLAMA_CPP_DIR / "llama-quantize",
    ])

    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p

    # Fall back to PATH (some distros ship llama.cpp binaries).
    on_path = shutil.which("llama-quantize")
    return Path(on_path) if on_path else None


def preflight(repo_id: str, dest_dir: Path, quantize: bool) -> tuple[bool, str]:
    """Check disk space and llama.cpp availability before a convert.

    Returns (ok, message). Message is a diagnostic string either way.
    """
    script = find_convert_script()
    if script is None:
        return (
            False,
            "llama.cpp convert script not found. Run 'anvil models convert-setup' "
            "or set LLAMA_CPP_PATH to your llama.cpp checkout.",
        )
    if quantize and find_quantize_binary() is None:
        return (
            False,
            "llama-quantize binary not found. Build llama.cpp "
            f"(cd {DEFAULT_LLAMA_CPP_DIR} && cmake -B build && cmake --build build --config Release).",
        )

    # Disk check: best-effort, since we don't know the real model size until
    # HF download starts. Use a conservative floor of 20 GB free.
    try:
        stat = shutil.disk_usage(dest_dir.parent if not dest_dir.exists() else dest_dir)
        free_gb = stat.free / (1024 ** 3)
    except OSError as e:
        return False, f"Cannot check disk space for {dest_dir}: {e}"
    if free_gb < 20:
        return False, f"Only {free_gb:.1f} GB free at {dest_dir}. Need at least 20 GB for a small model."

    return True, f"OK. {free_gb:.1f} GB free, convert script at {script}."


def _download_safetensors(
    repo_id: str,
    dest_dir: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """Download safetensors + tokenizer from a HF repo via huggingface_hub.

    Raises RuntimeError if huggingface_hub is not installed.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub not installed. Install with: pip install -e '.[convert]'"
        ) from e

    if progress_cb:
        progress_cb(f"Downloading {repo_id} from Hugging Face…")

    # Only fetch what the convert script actually needs. Skip GGUF files (we're
    # converting from safetensors), and skip ONNX/PyTorch binaries that some
    # repos also publish.
    allow = [
        "*.safetensors",
        "*.json",
        "tokenizer*",
        "*.model",
        "*.tiktoken",
        "*.txt",
    ]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest_dir),
        allow_patterns=allow,
        token=token,
    )
    if progress_cb:
        progress_cb(f"Downloaded to {path}")
    return Path(path)


def _run_convert_script(
    src_dir: Path,
    out_path: Path,
    convert_script: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> None:
    """Run llama.cpp's convert_hf_to_gguf.py. Raises CalledProcessError on failure."""
    cmd = [
        sys.executable,
        str(convert_script),
        str(src_dir),
        "--outfile", str(out_path),
        "--outtype", "f16",
    ]
    log.info("Running: %s", " ".join(cmd))
    if progress_cb:
        progress_cb(f"Converting to f16 GGUF: {out_path.name}…")
    subprocess.run(cmd, check=True)


def _run_quantize(
    src_gguf: Path,
    dst_gguf: Path,
    quant: str,
    quantize_binary: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> None:
    """Run llama-quantize. Raises CalledProcessError on failure."""
    cmd = [str(quantize_binary), str(src_gguf), str(dst_gguf), quant]
    log.info("Running: %s", " ".join(cmd))
    if progress_cb:
        progress_cb(f"Quantizing to {quant}: {dst_gguf.name}…")
    subprocess.run(cmd, check=True)


def _write_modelfile(gguf_path: Path, modelfile_path: Path, template: str | None = None) -> None:
    """Write an Ollama Modelfile pointing at a GGUF file."""
    lines = [f"FROM {gguf_path.resolve()}"]
    if template:
        lines.append(f'TEMPLATE """{template}"""')
    modelfile_path.write_text("\n".join(lines) + "\n")


def _ollama_create(name: str, modelfile_path: Path, progress_cb: Callable[[str], None] | None = None) -> None:
    """Register a GGUF with Ollama via ``ollama create``. Raises CalledProcessError on failure."""
    cmd = ["ollama", "create", name, "-f", str(modelfile_path)]
    log.info("Running: %s", " ".join(cmd))
    if progress_cb:
        progress_cb(f"Registering with Ollama as '{name}'…")
    subprocess.run(cmd, check=True)


def convert_hf_repo(
    repo_id: str,
    *,
    output_dir: Path,
    quant: str | None = "Q4_K_M",
    ollama_name: str | None = None,
    template: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
    register_with_ollama: bool = True,
    keep_safetensors: bool = False,
) -> ConvertResult:
    """Full pipeline: HF safetensors -> f16 GGUF -> quantized GGUF -> Ollama.

    Args:
        repo_id: ``org/repo`` on Hugging Face.
        output_dir: Where to place the safetensors checkout and GGUF outputs.
            Set this to a shared folder (e.g., a Dropbox directory) to make
            the resulting GGUF reachable from other machines.
        quant: Target quantization (e.g., ``Q4_K_M``). ``None`` keeps f16.
        ollama_name: Name to register under. Defaults to the repo's short name.
        template: Optional Ollama chat template. Most modern llama.cpp GGUFs
            embed a template, so leave this None unless you want to override.
        progress_cb: Called with human-readable status strings.
        register_with_ollama: If False, stop after producing the GGUF file —
            useful when converting on one machine for use on another.
        keep_safetensors: If False (default), delete the safetensors checkout
            after conversion to free disk.

    Returns:
        ``ConvertResult`` with the final GGUF path and Ollama name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ok, msg = preflight(repo_id, output_dir, quantize=quant is not None)
    if not ok:
        return ConvertResult(ok=False, message=msg)

    convert_script = find_convert_script()
    assert convert_script is not None  # preflight guarantees this

    safetensors_dir = output_dir / repo_id.replace("/", "--")
    short_name = repo_id.split("/")[-1]
    f16_gguf = output_dir / f"{short_name}.f16.gguf"
    final_gguf = output_dir / f"{short_name}.{quant}.gguf" if quant else f16_gguf
    target_name = ollama_name or short_name.lower().replace("-gguf", "").replace("_", "-")

    try:
        _download_safetensors(repo_id, safetensors_dir, progress_cb)
    except Exception as e:
        return ConvertResult(ok=False, message=f"Download failed: {e}")

    try:
        _run_convert_script(safetensors_dir, f16_gguf, convert_script, progress_cb)
    except subprocess.CalledProcessError as e:
        return ConvertResult(ok=False, message=f"GGUF conversion failed (exit {e.returncode}).")
    except Exception as e:
        return ConvertResult(ok=False, message=f"GGUF conversion failed: {e}")

    if quant is not None:
        quantize_binary = find_quantize_binary()
        if quantize_binary is None:
            return ConvertResult(
                ok=False,
                message="Quantize binary disappeared between preflight and run.",
                gguf_path=f16_gguf,
            )
        try:
            _run_quantize(f16_gguf, final_gguf, quant, quantize_binary, progress_cb)
        except subprocess.CalledProcessError as e:
            return ConvertResult(
                ok=False,
                message=f"Quantize failed (exit {e.returncode}).",
                gguf_path=f16_gguf,
            )
        # f16 is the largest intermediate; remove it once we have the quantized copy.
        try:
            f16_gguf.unlink()
        except OSError as e:
            log.debug("Could not remove intermediate f16: %s", e)

    if not keep_safetensors:
        shutil.rmtree(safetensors_dir, ignore_errors=True)

    if not register_with_ollama:
        return ConvertResult(
            ok=True,
            message=f"GGUF ready at {final_gguf}. Import elsewhere with 'anvil models import'.",
            gguf_path=final_gguf,
        )

    modelfile = output_dir / f"{target_name}.Modelfile"
    _write_modelfile(final_gguf, modelfile, template)
    try:
        _ollama_create(target_name, modelfile, progress_cb)
    except subprocess.CalledProcessError as e:
        return ConvertResult(
            ok=False,
            message=f"ollama create failed (exit {e.returncode}).",
            gguf_path=final_gguf,
        )

    return ConvertResult(
        ok=True,
        message=f"Registered as '{target_name}'. Use: anvil chat --model {target_name}",
        gguf_path=final_gguf,
        ollama_name=target_name,
    )


def import_gguf(
    gguf_path: Path,
    *,
    name: str | None = None,
    template: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> ConvertResult:
    """Register an existing local GGUF with Ollama.

    Useful when a GGUF was produced on another machine (e.g., the M5) and
    moved to the current host via Dropbox/rsync/sneakernet.
    """
    gguf_path = gguf_path.expanduser().resolve()
    if not gguf_path.is_file():
        return ConvertResult(ok=False, message=f"GGUF file not found: {gguf_path}")
    if gguf_path.suffix.lower() != ".gguf":
        return ConvertResult(ok=False, message=f"Not a .gguf file: {gguf_path}")

    target_name = name or gguf_path.stem.lower().replace(".", "-").replace("_", "-")
    modelfile = gguf_path.parent / f"{target_name}.Modelfile"
    _write_modelfile(gguf_path, modelfile, template)

    try:
        _ollama_create(target_name, modelfile, progress_cb)
    except FileNotFoundError:
        return ConvertResult(ok=False, message="'ollama' CLI not found on PATH.")
    except subprocess.CalledProcessError as e:
        return ConvertResult(ok=False, message=f"ollama create failed (exit {e.returncode}).")

    return ConvertResult(
        ok=True,
        message=f"Registered as '{target_name}'. Use: anvil chat --model {target_name}",
        gguf_path=gguf_path,
        ollama_name=target_name,
    )


def setup_llama_cpp(
    dest: Path = DEFAULT_LLAMA_CPP_DIR,
    build_quantize: bool = True,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Clone llama.cpp and optionally build the quantize binary.

    Returns (ok, message). Idempotent: if the repo is already cloned we
    just pull latest; if the binary is already built we skip the build.
    """
    dest = dest.expanduser()
    if progress_cb:
        progress_cb(f"Setting up llama.cpp at {dest}…")

    try:
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(dest)], check=True)
        else:
            subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
    except FileNotFoundError:
        return False, "'git' not found on PATH."
    except subprocess.CalledProcessError as e:
        return False, f"git clone/pull failed (exit {e.returncode})."

    if not build_quantize:
        return True, f"Cloned to {dest} (binary not built — pass build_quantize=True to build)."

    if find_quantize_binary(dest) is not None:
        return True, f"Already built. Binary at {find_quantize_binary(dest)}."

    if progress_cb:
        progress_cb("Building llama-quantize (this takes a few minutes)…")
    try:
        subprocess.run(["cmake", "-B", "build", "-S", str(dest)], check=True, cwd=dest)
        subprocess.run(
            ["cmake", "--build", "build", "--config", "Release", "--target", "llama-quantize", "-j"],
            check=True,
            cwd=dest,
        )
    except FileNotFoundError:
        return False, (
            "'cmake' not found on PATH. Install with one of:\n"
            "  pip install cmake   (no sudo; installs into the active venv)\n"
            "  sudo apt install cmake   (Debian/Ubuntu)\n"
            "  brew install cmake   (macOS)"
        )
    except subprocess.CalledProcessError as e:
        return False, f"llama.cpp build failed (exit {e.returncode})."

    binary = find_quantize_binary(dest)
    if binary is None:
        return False, "Build finished but llama-quantize binary not found."
    return True, f"Ready. Binary at {binary}."
