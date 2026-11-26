import json
import tarfile
import zipfile
from pathlib import Path

from packaging.requirements import Requirement

import edsnlp
from eds_pseudo.cli import package

ROOT = Path(__file__).parents[1]


def test_dependency_package_contains_loader_without_architecture(tmp_path):
    nlp = edsnlp.blank("eds")
    # A run folder with the metric file of one evaluation, as written by `eds-pseudo evaluate`
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prf = {"p": 0.9, "r": 0.8, "f": 0.85}
    scores = {
        "exact": {"micro": prf},
        "token": {"NOM": prf, "PRENOM": prf, "micro": prf},
        "redact": {"micro": {"token_r": 0.95, "full": 0.5, "documents": 688}},
    }
    (run_dir / "original-test-metrics.json").write_text(json.dumps({"final": scores}))
    package(
        pipeline=nlp,
        name="eds-pseudo-public",
        root_dir=ROOT,
        dist_dir=tmp_path,
        distributions=["wheel", "sdist"],
        code_check="off",
        isolation=False,
        card_metadata={"license": "bsd-3-clause"},
        results_dir=run_dir,
        results={"original-test": "AP-HP test notes"},
        label_results=["original-test"],
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
        readme = metadata[metadata.index("\n\n") + 2 :]
        assert readme.startswith("---\nlicense: bsd-3-clause\nmodel-index:\n")
        assert "name: Token Scores / NOM / F1" in readme
        assert "| AP-HP test notes | 688 | 0.900 | 0.800 | 0.850 | 0.950 | 0.500 |" in readme
        assert "<!-- metrics -->" not in readme
    with tarfile.open(source) as archive:
        names = archive.getnames()
        assert any(name.endswith("/eds_pseudo_public/__init__.py") for name in names)
        assert not any("/eds_pseudo/" in name for name in names)
