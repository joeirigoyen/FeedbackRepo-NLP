import subprocess
from pathlib import Path


if __name__ == '__main__':
    CWD = Path.cwd()
    FILE_PATH = CWD.joinpath("scripts", "flow_tester.py")
    cmd = f"python \"{FILE_PATH}\""
    print(cmd)
    subprocess.call(cmd, shell=True)
