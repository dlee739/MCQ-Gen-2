from __future__ import annotations

import tomllib
from pathlib import Path

from mcqgen2 import __version__

ROOT = Path(__file__).parents[1]


def test_package_and_beta_versions_match() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == __version__ == "0.1.0b1"


def test_example_environment_does_not_contain_a_key() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    setting = next(line for line in lines if line.startswith("OPENAI_API_KEY="))
    assert setting == "OPENAI_API_KEY="


def test_beta_launcher_and_release_notes_are_present() -> None:
    assert (ROOT / "setup_beta.bat").is_file()
    assert (ROOT / "run_app.bat").is_file()
    assert (ROOT / "BETA_RELEASE_NOTES.md").is_file()
