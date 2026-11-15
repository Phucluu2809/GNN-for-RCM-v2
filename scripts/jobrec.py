import os
import subprocess
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    native = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python311" / "python.exe"
    if os.name == "nt" and "mingw" in sysconfig.get_platform().lower() and native.is_file():
        return subprocess.call([str(native), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    from jobrec.cli import main as cli_main
    cli_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
