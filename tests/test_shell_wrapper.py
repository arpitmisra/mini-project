"""Exercise actual Bash dispatch without installing packages or running a model."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from oct_dinov2_mlx.config import ROOT


@pytest.mark.parametrize("arguments,environment", [
    (["inspect", "scan with spaces.jpg"], ".venv-dense"),
    (["inspect", "--backend", "mlx"], ".venv-mlx-cpu"),
    (["inspect", "--backend=mlx"], ".venv-mlx-cpu"),
    (["finetune", "dataset with spaces", "--mode", "lora"], ".venv-mlx-cpu"),
    (["inspect", "scan --backend mlx image.jpg"], ".venv-dense"),
])
def test_wrapper_dispatch_preserves_arguments(tmp_path, arguments, environment):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else None
    bash = str(bash) if bash and bash.is_file() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash unavailable")
    shutil.copyfile(ROOT / "oct.sh", tmp_path / "oct.sh")
    for name in (".venv-dense", ".venv-mlx-cpu"):
        python = tmp_path / name / "bin/python"
        python.parent.mkdir(parents=True)
        python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$0" "$@"\n', encoding="utf-8")
        python.chmod(0o755)
    completed = subprocess.run([bash, (tmp_path / "oct.sh").as_posix(), *arguments],
                               capture_output=True, text=True, check=True, timeout=20)
    lines = completed.stdout.splitlines()
    assert environment in lines[0]
    assert lines[1:] == ["oct.py", *arguments]
