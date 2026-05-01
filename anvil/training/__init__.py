"""LoRA / QLoRA training stack for ROCm + unified-memory APUs.

Wired to the CLI as `anvil train {install,setup,preflight,diagnose,run}`.
See docs/qlora-training.md for the validated install path on Strix Halo
(gfx1151) and the four traps to avoid.
"""

from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent / "scripts"


def script_path(name: str) -> Path:
    """Resolve a bundled .sh script by basename (e.g. 'preflight-training.sh')."""
    p = SCRIPTS_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"bundled script not found: {p}")
    return p
