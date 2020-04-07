import os
import torch
import logging as log


class Trainer:
    def __init__(
        self,
        model,
        task,
        bs,
        lr,
        weight_decay,
        max_epochs,
        warmup_iters_ratio,
        val_interval_iters,
        report_interval_iters,
        stopping_patience,
        exp_dir,
    ):
        self.task = task
        self.model = model
        self.task.preprocess_data(model=model)
        self.task.build_iterators(bs=bs)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=max_epochs,
            steps_per_epoch=len(task.iterators["train"]),
            pct_start=warmup_iters_ratio,
            anneal_strategy="linear",
            cycle_momentum=False,
        )

        self.device = None
        self.amp = None

        self.max_epochs = max_epochs
        self.val_interval_iters = val_interval_iters
        self.report_interval_iters = report_interval_iters
        self.stopping_patience = stopping_patience
        self.exp_dir = exp_dir

        training_steps = len(self.task.iterators["train"])
        if training_steps < self.val_interval_iters:
            log.info(f"val_interval too large. override to {training_steps}")
            self.val_interval_iters = training_steps
        if training_steps < self.report_interval_iters:
            log.info(f"report_interval too large. override to {training_steps}")
            self.report_interval_iters = training_steps

    def to(self, device, amp):
        self.device = device
        self.amp = amp

    def move_inputs_to_device(self, batch_inputs):
        for key, value in batch_inputs.items():
            if isinstance(value, torch.Tensor):
                batch_inputs[key] = value.to(self.device)
        return batch_inputs

    def train(self):
        log.info("start training")
        if self.amp:
            from apex import amp

        self.model.train()
        training_results = {"best_acc": 0.0, "best_iter": -1, "current_iter": 0}
        score_record = {"acc": [], "count": []}

        for epoch in range(self.max_epochs):
            log.info(f"train epoch {epoch + 1} / {self.max_epochs}")

            for batch, batch_inputs in enumerate(self.task.iterators["train"]):
                self.model.zero_grad()
                batch_outputs = self.model(self.move_inputs_to_device(batch_inputs))
                if self.amp:
                    with amp.scale_loss(batch_outputs["loss"], self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    batch_outputs["loss"].backward()
                score_record["acc"].append(batch_outputs["acc"].item())
                score_record["count"].append(len(batch_inputs["uid"]))

                self.optimizer.step()
                self.scheduler.step()

                training_results["current_iter"] += 1
                if training_results["current_iter"] % self.report_interval_iters == 0:
                    average_acc = sum(
                        [a * c for a, c in zip(score_record["acc"], score_record["count"])]
                    ) / sum(score_record["count"])
                    log.info(
                        f'train batch {batch + 1} / {len(self.task.iterators["train"])}'
                        f'(iter {training_results["current_iter"]}),'
                        f"current average acc {average_acc}"
                    )
                    score_record = {"acc": [], "count": []}
                if training_results["current_iter"] % self.val_interval_iters == 0:
                    val_acc = self.eval("val")["acc"]
                    log.info(f"val acc {val_acc}")
                    if val_acc > training_results["best_acc"]:
                        training_results["best_acc"] = val_acc
                        training_results["best_iter"] = training_results["current_iter"]
                        log.info(f"best val acc updated\n{training_results}")
                        self.save_model(os.path.join(self.exp_dir, "best_model.pt"))
                    elif (
                        training_results["current_iter"]
                        > training_results["best_iter"]
                        + self.val_interval_iters * self.stopping_patience
                    ):
                        log.info("out of patience")
                        break

        log.info(f"training done\n{training_results}")
        return training_results

    def eval(self, split):
        log.info(f"start evaluating on {split}")
        self.model.eval()
        eval_results = {"acc": None, "label_pred": []}
        score_record = {"acc": [], "count": []}

        with torch.no_grad():
            for batch, batch_inputs in enumerate(self.task.iterators[split]):
                batch_outputs = self.model(self.move_inputs_to_device(batch_inputs))
                score_record["acc"].append(batch_outputs["acc"].item())
                score_record["count"].append(len(batch_inputs["uid"]))
                eval_results["label_pred"].append(batch_outputs["label_pred"].tolist())

                if (batch + 1) % self.report_interval_iters == 0:
                    average_acc = sum(
                        [a * c for a, c in zip(score_record["acc"], score_record["count"])]
                    ) / sum(score_record["count"])
                    log.info(
                        f"eval batch {batch + 1} / {len(self.task.iterators[split])},"
                        f"current average acc {average_acc}"
                    )

        eval_results["acc"] = sum(
            [a * c for a, c in zip(score_record["acc"], score_record["count"])]
        ) / sum(score_record["count"])

        log.info(f"validating done")
        self.model.train()
        return eval_results

    def load_model(self, model_ckpt):
        log.info(f"load model from {model_ckpt}")
        self.model.load_state_dict(torch.load(model_ckpt, map_location=self.device), strict=False)

    def save_model(self, model_ckpt):
        log.info(f"save model to {model_ckpt}")
        torch.save(self.model.state_dict(), model_ckpt)
