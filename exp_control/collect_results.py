import json
import pandas as pd
import os
import argparse

from shared_settings import decode_exp_name


def collect_results(args):
    results = {
        "dataset": [],
        "framing": [],
        "learning_rate": [],
        "batch_size": [],
        "max_epochs": [],
        "seed": [],
        "best_val_accuracy": [],
        "exp_name": [],
    }

    def record_exp(one_exp_result):
        dataset, framing, lr, bs, max_epochs, seed = decode_exp_name(one_exp_result["exp_name"])
        results["dataset"].append(dataset)
        results["framing"].append(framing)
        results["learning_rate"].append(lr)
        results["batch_size"].append(bs)
        results["max_epochs"].append(max_epochs)
        results["seed"].append(seed)
        results["best_val_accuracy"].append(one_exp_result["best_acc"])
        results["exp_name"].append(one_exp_result["exp_name"])

    with open(os.path.join(args.results_dir, "val_summary.jsonl"), "r") as reader:
        for row in reader:
            one_exp_result = json.loads(row)
            record_exp(one_exp_result)

    df_raw = pd.DataFrame.from_dict(results)
    df_raw.sort_values(by=["dataset", "best_val_accuracy"], ascending=False, inplace=True)
    df_raw.to_csv(os.path.join(args.results_dir, "raw_results.csv"), index=False)

    # TODO:
    # 1. a tsv file of all experiments, eliminate seed, summarize best_val_accuracy to p0.25,
    # sorted by (dataset, framing, val_accuracy)

    # 2. a tsv file of only the best HP in each (dataset, framing).

    return


if __name__ == "__main__":
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Collect results and make tsv")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.getenv("NLU_RESULTS_DIR", os.path.join(repo_dir, "results")),
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.getenv("NLU_DATA_DIR", os.path.join(repo_dir, "data"))
    )

    args = parser.parse_args()
    args.repo_dir = repo_dir
    collect_results(args)
