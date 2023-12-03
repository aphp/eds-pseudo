import os
import shutil
import sys

import pytest
from confit import Config
from confit.utils.random import set_seed

from eds_pseudo.adapter import PseudoReader
from eds_pseudo.scorer import PseudoScorer
from edsnlp import registry

sys.path.append("../scripts")

from scripts.package import package  # noqa: E402
from scripts.train import train  # noqa: E402


@pytest.fixture
def run_in_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.mark.parametrize(
    "batch_size,do_package", [("10 samples", True), ("50 words", False)]
)
def test_train(run_in_test_dir, tmp_path, batch_size, do_package):
    set_seed(42)
    config = Config.from_disk("../configs/config.cfg")
    config = config.merge(
        {
            "training_docs": {
                "source": {"path": "../data/gen_dataset/train.jsonl"},
                "limit": 10,
                "max_length": 50,
                "randomize": True,
            },
            "val_docs": {
                "source": {"path": "../data/gen_dataset/train.jsonl"},
                "limit": 10,
            },
            "components": {
                "embedding": {"embedding": {"model": "hf-internal-testing/tiny-bert"}}
            },
            "train": {
                "max_steps": 10,
                "batch_size": batch_size,
                "validation_interval": 5,
                "grad_accumulation_max_tokens": 10,
                "cpu": True,
            },
        }
    )
    shutil.rmtree(tmp_path, ignore_errors=True)
    kwargs = config["train"].resolve(registry=registry, root=config)
    nlp = train(**kwargs, output_dir=tmp_path)
    scorer = PseudoScorer(**kwargs["scorer"])
    last_scores = scorer(nlp, list(PseudoReader(**kwargs["val_data"])(nlp)))

    assert "f" in last_scores

    if do_package:
        os.chdir("..")
        # Will use [tool.edsnlp].model_name in pyproject.toml
        package(
            model=tmp_path / "model-last",
            build_dir=tmp_path / "build",
            dist_dir=tmp_path / "dist",
        )
        os.chdir("tests")

        assert len(list((tmp_path / "dist").iterdir())) == 1
