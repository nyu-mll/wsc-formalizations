import os
import logging as log
import sys
import json
from apex import amp


from args import parser, check_config
from models import WSCVariantModel
from tasks import WSCLikeTask
from trainer import Trainer
from utils import config_logging


def main():
    # parse arguments
    cfg = parser.parse_args()
    check_config(cfg)

    # setup logging
    config_logging(os.path.join(cfg.exp_dir, f"{cfg.mode}.log"))
    log.info(f"Experiment {cfg.exp_name}_{cfg.mode}")

    # setup results
    cfg.exp_dir = os.path.join(cfg.results_dir, cfg.exp_name)
    cfg.cache_dir = os.path.join(cfg.results_dir, "cache")
    os.makedirs(cfg.exp_dir)
    if not os.path.exists(cfg.exp_dir):
        os.makedirs(cfg.exp_dir)
    if not os.path.exists(cfg.cache_dir):
        os.makedirs(cfg.cache_dir)

    # create task, model and trainer
    task = WSCLikeTask(
        framing=cfg.framing,
        dataset=cfg.dataset,
        data_dir=cfg.data_dir,
        cache_dir=cfg.cache_dir,
        reload_data=cfg.reload_data,
    )
    model = WSCVariantModel(framing=cfg.framing, pretrained=cfg.pretrained, cache_dir=cfg.cache_dir)
    trainer = Trainer(model=model, task=task, exp_dir=cfg.exp_dir, cfg=cfg)

    # setup device
    model.to(cfg.device)
    trainer.to(cfg.device, cfg.amp)
    if cfg.amp:
        model, optimizer = amp.initialize(model, trainer.optimizer, opt_level="O1")

    # load model
    if cfg.load_model_state != "":
        trainer.load_model(cfg.load_model_state)

    # run trainer
    if cfg.mode == "train":
        result_dict = trainer.train()
        result_dict["exp_name"] = cfg.exp_name
        with open(os.path.join(cfg.results_dir, "val_summary.jsonl"), "a") as f:
            f.write(json.dumps(result_dict))
    elif cfg.mode == "eval":
        pred = trainer.eval(split="test")["query_pred"]
        task.write_pred(pred=pred, filename=os.path.join(cfg.results_dir, f"{cfg.exp_name}.submit"))

    sys.exit(0)


if __name__ == "__main__":
    main()
