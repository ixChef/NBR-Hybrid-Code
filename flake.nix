{
  description = "Generic nix dev shell that installs requirements.txt into a venv (wheels only)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config = { allowUnfree = true; };
      };

      python = pkgs.python311;

      baseLibs = with pkgs; [
        stdenv.cc.cc.lib
        zlib
        glib
        libglvnd
        libGL
        libxcb
        libx11
        libxext
        libxrender
        libxi
        libxfixes
        openssl
      ];

      ldPath = pkgs.lib.makeLibraryPath baseLibs;
    in {
      devShells.${system}.default = pkgs.mkShell {
        name = "Temporal FYP Flake";

        packages = [
          python
          pkgs.python311Packages.pip
          pkgs.python311Packages.setuptools
          pkgs.python311Packages.wheel
          pkgs.python311Packages.annoy

          pkgs.git
          pkgs.sqlite
          
          
          pkgs.gcc
          pkgs.gnumake
          pkgs.pkg-config
          pkgs.openblas
        ] ++ baseLibs;

        shellHook = ''
          set -e

          echo "[shell] Entering dev shell"

          export LD_LIBRARY_PATH="${ldPath}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
          export PIP_PREFER_BINARY=1
          export PIP_ONLY_BINARY=":all:"
          export PIP_DISABLE_PIP_VERSION_CHECK=1

          if [ ! -d .venv ]; then
            echo "[shell] Creating virtualenv in .venv"
            ${python.interpreter} -m venv --system-site-packages .venv
          fi

          . .venv/bin/activate

          python - <<'EOF'
import hashlib
import os
import pathlib
import subprocess
import sys

reqPath = pathlib.Path("requirements.txt")
venvPath = pathlib.Path(".venv")
stampPath = venvPath / ".requirements.sha256"
filteredPath = venvPath / ".requirements.filtered.txt"

if not reqPath.exists():
    print("[shell] requirements.txt not found; skipping dependency install.")
    raise SystemExit(0)

reqText = reqPath.read_text(encoding="utf-8", errors="ignore")
sha = hashlib.sha256(reqText.encode("utf-8")).hexdigest()

if stampPath.exists() and stampPath.read_text().strip() == sha:
    print("[shell] requirements.txt unchanged; deps already satisfied.")
    raise SystemExit(0)

lines = []
for line in reqText.splitlines():
    s = line.strip()
    if not s or s.startswith("#"):
        continue
    if s.lower().startswith("annoy"):
        continue
    lines.append(s)

filteredPath.write_text("\n".join(lines) + "\n", encoding="utf-8")

if "+cu118" in reqText:
    os.environ["PIP_EXTRA_INDEX_URL"] = "https://download.pytorch.org/whl/cu118"

subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--prefer-binary",
    "--only-binary", ":all:",
    "--upgrade-strategy", "only-if-needed",
    "-r", str(filteredPath),
])

stampPath.write_text(sha)
EOF

          export PATH="$PWD/.venv/bin:$PATH"
          export PYTHONPATH="$PWD/src:$PYTHONPATH"

          echo "[shell] dev shell ready. Python: $(python --version)"
        '';
      };
    };
}
