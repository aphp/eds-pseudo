import argparse
import subprocess

parser = argparse.ArgumentParser("experiments")
parser.add_argument("idx", help="Process index", nargs="?", type=int, default=None)
args = parser.parse_args()
process_idx = args.idx

# Automated grid search experiments
seeds = [42, 43, 44, 45, 46]
limits = [10, 30, 60, 100, 150, 250, 500, 700, 1000, 1300, 1700, 2000, 2500, 3000, 0]
bert_names = [
    "/export/home/pwajsburt/data/models/embedding-whole-word/checkpoint-250000/",
    "/export/home/share/datascientists/models/camembert-base",
    "/export/home/share/datascientists/models/training-from-scratch-2021-08-13/",
]
doc_ablations = [
    "'doc._.note_class_source_value != \"CR-ACTE-DIAG-AUTRE\"'",
    "'doc._.note_class_source_value != \"CR-ANAPATH\"'",
    "'doc._.note_class_source_value != \"CR-IMAGE\"'",
    "'doc._.note_class_source_value != \"CR-OPER\"'",
    "'doc._.note_class_source_value != \"RCP\"'",
]

# Iterate over all combinations of hyperparameter values.
for seed in seeds:
    # Iterate over all combinations of hyperparameter values.
    for expr in doc_ablations:
        xp = [
            "dvc",
            "exp",
            "run",
            "--queue",
            "-S",
            f"configs/config.cfg:system.seed={seed}",
            "-S",
            f"configs/config.cfg:corpora.train.filter_expr={expr}",
        ]
        if process_idx is None:
            print("Running", " ".join(xp))
            subprocess.run(xp)

    for bert_name in bert_names:
        xp = [
            "dvc",
            "exp",
            "run",
            "--queue",
            "-S",
            f"configs/config.cfg:system.seed={seed}",
            "-S",
            f"configs/config.cfg:paths.bert={bert_name}",
        ]
        if process_idx is None:
            print("Running", " ".join(xp))
            subprocess.run(xp)

    ## Iterate over all combinations of hyperparameter values.
    for limit in limits:
        if limit == 0:
            # We already performed the xp with no limit above
            continue
        xp = [
            "dvc",
            "exp",
            "run",
            "--queue",
            "-S",
            f"configs/config.cfg:system.seed={seed}",
            "-S",
            f"configs/config.cfg:corpora.train.limit={limit}",
        ]
        if process_idx is None:
            print("Running", " ".join(xp))
            subprocess.run(xp)
