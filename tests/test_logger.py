from pathlib import Path

from eds_pseudonymisation.utils.loggers import dvc_live_logger


def test_logger(tmp_path: Path):
    log_step, finalize = dvc_live_logger(
        path=tmp_path,
        categories=["NOM"],
    )(None)

    log_step(
        {
            "epoch": 0,
            "step": 0,
            "score": 0.0,
            "losses": {
                "textcat": 1.0,
            },
            "other_scores": {
                "cats_f_per_type": {
                    "NOM": {
                        "p": 0.5,
                        "r": 1.0,
                        "f": 0.66666,
                    },
                },
            },
        }
    )

    finalize()
