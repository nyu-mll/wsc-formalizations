"""
Hyper parameter tuning based on RoBERTa on WSC
"""

from random import randrange, random
import os
import argparse
from shared_settings import make_command

lr_candidates = [1e-5, 2e-5, 3e-5]
bs_candidates = [8, 16, 32, 64]
max_epochs_candidates = [5, 10, 20, 40]
seed_range = 1e6


def select_candidates():
    lr = lr_candidates[randrange(0, len(lr_candidates), 1)]
    bs = bs_candidates[randrange(0, len(bs_candidates), 1)]
    max_epochs = max_epochs_candidates[randrange(0, len(max_epochs_candidates), 1)]
    seed = random.randint(0, seed_range)

    return lr, bs, max_epochs, seed


def submit_trials(args):
    jobs = []

    for trial in range(args.n_trials):
        # select candidates for trial
        lr, bs, max_epochs, seed = select_candidates()
        command = make_command(
            args.dataset, args.framing, lr, bs, max_epochs, seed, args.gpu_capacity
        )
        sbatch_file = os.path.join(args.repo_dir, "exp_control", f"{args.user}.sbatch")
        jobs.append(f'COMMAND="{command}" sbatch {sbatch_file}')

    with open("submit_sbatch.sh", "w") as f:
        f.writelines(jobs)

    return


if __name__ == "__main__":
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Run HP Search")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.getenv("NLU_RESULTS_DIR", os.path.join(repo_dir, "results")),
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.getenv("NLU_DATA_DIR", os.path.join(repo_dir, "data"))
    )
    parser.add_argument("--user", type=str)

    parser.add_argument("--n-trials", type=int, help="number of trials")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wsc-cross",
        choices=[
            "wsc-spacy",
            "wsc-cross",
            "winogrande-xs",
            "winogrande-s",
            "winogrande-m",
            "winogrande-l",
            "winogrande-xl",
        ],
    )
    parser.add_argument(
        "--framing",
        type=str,
        choices=[
            "P-SPAN",
            "P-SENT",
            "MC-SENT-PLOSS",
            "MC-SENT-PAIR",
            "MC-SENT-SCALE",
            "MC-SENT",
            "MC-MLM",
        ],
    )
    parser.add_argument("--gpu-capacity", type=int, default=8)

    args = parser.parse_args()
    args.repo_dir = repo_dir
    submit_trials(args)
