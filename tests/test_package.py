import tarfile
import zipfile
from pathlib import Path

from packaging.requirements import Requirement

import edsnlp
from eds_pseudo.cli import package

ROOT = Path(__file__).parents[1]


def test_dependency_package_contains_loader_without_architecture(tmp_path):
    nlp = edsnlp.blank("eds")
    package(
        pipeline=nlp,
        name="eds-pseudo-public",
        root_dir=ROOT,
        dist_dir=tmp_path,
        distributions=["wheel", "sdist"],
        code_check="off",
        isolation=False,
        readme_replacements={"<!-- metrics -->": "## Public model metrics"},
    )

    wheel = tmp_path / "eds_pseudo_public-1.0.0-py3-none-any.whl"
    source = tmp_path / "eds_pseudo_public-1.0.0.tar.gz"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "eds_pseudo_public/__init__.py" in names
        assert "eds_pseudo_public/artifacts/config.cfg" in names
        assert not any(name.startswith("eds_pseudo/") for name in names)
        metadata = archive.read("eds_pseudo_public-1.0.0.dist-info/METADATA").decode()
        requirements = {
            Requirement(line.removeprefix("Requires-Dist: "))
            for line in metadata.splitlines()
            if line.startswith("Requires-Dist: ")
        }
        assert Requirement("eds-pseudo>=1,<2") in requirements
        assert "## Public model metrics" in metadata
    with tarfile.open(source) as archive:
        names = archive.getnames()
        assert any(name.endswith("/eds_pseudo_public/__init__.py") for name in names)
        assert not any("/eds_pseudo/" in name for name in names)
