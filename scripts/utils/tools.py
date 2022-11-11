import shlex
import subprocess
from pathlib import Path


def default_path(filepath: str | Path) -> Path:
    return Path(filepath) if isinstance(filepath, str) else filepath


def get_text_lines(filepath: str | Path) -> list:
    with open(filepath) as datasource:
        return datasource.readlines()


def decode_output(cmd: str, encoding: str = "utf-8") -> tuple[str, str]:
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    return out.decode(encoding), err.decode(encoding)
