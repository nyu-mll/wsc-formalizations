import argparse
import os

parser = argparse.ArgumentParser(description="WSC Trick Experiments")

# basic settings
# experiment name
parser.add_argument("--exp-name", type=str, default="debug")
# directory to save results, cached data, model states
parser.add_argument("--results-dir", type=str, default=os.getenv("NLU_DATA_DIR", "data/"))
# directory in which data are stored
parser.add_argument("--data-dir", type=str, default=os.getenv("NLU_RESULT_DIR", "results/"))
# mode
parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
# load model parameter before training / evaluating
parser.add_argument("--load-model-ckpt", type=str, default="")

# device settings
# device
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
# mixed precision
parser.add_argument("--amp", type=bool, action="store_false")


# data settings
# neglect cached data and reload
parser.add_argument("--reload-data", type=bool, action="store_true")
# dataset
parser.add_argument("--dataset", type=str, choices=["wsc", "winogrande"])
# framing
parser.add_argument(
    "--framing",
    type=str,
    choices=[
        "P-SPAN",
        "P-SENT",
        "MC-SENT-PLOSS",
        "MC-SENT-NOSCALE",
        "MC-SENT-NOPAIR",
        "MC-SENT",
        "MC-MLM",
    ],
)

# training settings
# batch size
parser.add_argument("--bs", type=int, default=32)
# learning rate
parser.add_argument("--lr", type=float, default=1e-5)
# weight decay
parser.add_argument("--weight-decay", type=float, default=1e-3)
# number of epochs
parser.add_argument("--max-epochs", type=int, default=10)
# ratio of warmup iters to full training process
parser.add_argument("--warmup-iters-ratio", type=float, default=0.06)
# number of iterations between validation
parser.add_argument("--val-interval-iters", type=int, default=1000)
# number of iterations between reporting result
parser.add_argument("--report-interval-iters", type=int, default=200)
# number of validations waiting for better results before stopping, set to -1 to disable
parser.add_argument("--stopping-patience", type=int, default=-1)


# model settings
# which transformer model to use
parser.add_argument(
    "--pretrained",
    type=str,
    default="roberta-large",
    choices=["roberta-base", "roberta-large", "albert-base-v2", "albert-xxlarge-v2"],
)


def check_config(cfg):
    if cfg.device == "cpu":
        cfg.amp = False
