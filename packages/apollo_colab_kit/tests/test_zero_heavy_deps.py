"""Hard isolation test: importing apollo_colab_kit must pull ZERO heavy/MUSE deps."""

from __future__ import annotations

import subprocess
import sys


FORBIDDEN_MODULES = (
    "multimodal_gen",
    "pygame",
    "librosa",
    "numba",
    "llvmlite",
    "scipy",
    "numpy",
    "soundfile",
    "torch",
    "fluidsynth",
    "yt_dlp",
)


def test_fresh_interpreter_imports_zero_heavy_or_muse_modules():
    code = (
        "import sys\n"
        "import apollo_colab_kit\n"
        "import apollo_colab_kit.schemas\n"
        "import apollo_colab_kit.verifiers\n"
        "import apollo_colab_kit.prompt_sweep\n"
        "import apollo_colab_kit.dataset_export\n"
        f"forbidden = {FORBIDDEN_MODULES!r}\n"
        "loaded = [m for m in forbidden if any(k == m or k.startswith(m + '.') for k in sys.modules)]\n"
        "assert not loaded, f'Forbidden heavy/MUSE modules loaded: {loaded}'\n"
        "print('CLEAN_ISOLATION_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    assert "CLEAN_ISOLATION_OK" in proc.stdout
