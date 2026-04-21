#!/usr/bin/env bash
# ollama-anvil installer — sets up venv, installs package, verifies.
set -euo pipefail

say() { printf "\033[1;36m==> %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m!!  %s\033[0m\n" "$*"; }
die() { printf "\033[1;31mxx  %s\033[0m\n" "$*" >&2; exit 1; }

say "Checking Python..."
command -v python3 >/dev/null || die "python3 not found. Install Python 3.10+."
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_OK=$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')
[ "$PY_OK" = "1" ] || die "Python $PY_VER found, need 3.10+."
say "Python $PY_VER OK"

say "Checking Ollama..."
if ! command -v ollama >/dev/null; then
    warn "Ollama not found."
    echo "    Install: curl -fsSL https://ollama.com/install.sh | sh"
    echo "    (anvil will still install; add ollama later.)"
else
    ollama --version | sed 's/^/    /'
fi

say "Creating venv..."
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

say "Installing ollama-anvil..."
pip install --upgrade pip >/dev/null
pip install -e .

say "Verifying..."
anvil version

cat <<EOF

  Install complete.

  Next:
    source .venv/bin/activate
    anvil doctor
    anvil models recommend
    anvil                         # start chatting

EOF
