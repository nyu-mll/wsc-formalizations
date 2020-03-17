import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class RobertaMultihead(nn.Module):
    def __init__(self, framing, roberta_model="roberta-large"):
        """
        framing, one of
            "P-SPAN"
            "P-SENT"
            "MC-SENT-PLOSS"
            "MC-SENT-NOSCALE"
            "MC-SENT-NOPAIR"
            "MC-SENT"
            "MC-MLM"
        """
        super().__init__()

        self.padding_logits = -100

        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(roberta_model)
        self.mask_id = self.tokenizer.mask_token_id
        self.padding_id = self.tokenizer.pad_token_id

        transformer_with_lm = transformers.RobertaForMaskedLM.from_pretrained(roberta_model)
        self.hidden_size = transformer_with_lm.config.hidden_size
        self.framing = framing

        self.transformer = transformer_with_lm.roberta
        if "-SPAN-" in self.framing:
            self.span_head = nn.Linear(3 * self.hidden_size, 1)
        elif "-SENT-" in self.framing:
            self.sent_head = nn.Linear(self.hidden_size, 1)
        elif "-MLM-" in self.framing:
            self.mlm_head = transformer_with_lm.lm_head

    def forward(self, batch_inputs):
        """
        batch_inputs:
            for SPAN input
            "raw_input": (bs, seq_len)
            "span1_mask": (bs, seq_len)
            "span2_mask": (bs, seq_len)
            for SENT and MLM input
            "query_input": (bs, seq_len)
            "cand_input": (batch_cand_count, seq_len)
            for MLM input
            "mask_query_input": (bs, seq_len)
            "mask_cand_input": (batch_cand_count, seq_len)
            for MC prediction
            "cand_matching_idx": (bs, max_cands), between [-1, batch_cand_count)
            for P and MC prediction
            "label": (bs,)
        batch_outputs:
            "loss": (1,)
            "acc": (1,)
            "label_pred": (bs,)
        """
        if "-SPAN-" in self.framing:
            assert self.framing.startswith("P-")
            raw_repr = self.transformer(batch_inputs["raw_input"])
            cls_repr = raw_repr[:, 0]
            span1_mask = batch_inputs["span1_mask"].unsqueeze(dim=2)
            span1_repr = torch.sum(raw_repr * span1_mask, dim=1) / span1_mask.sum(dim=1)
            span2_repr = torch.sum(
                raw_repr * batch_inputs["span2_mask"].unsqueeze(dim=2), dim=1
            ) / batch_inputs["span2_mask"].sum(dim=1, keepdim=True)
            concat_repr = torch.cat([cls_repr, span1_repr, span2_repr], dim=2)
            query_logits = self.span_head(concat_repr)

        elif "-SENT-" in self.framing:
            query_repr = self.transformer(batch_inputs["query_input"])[:, 0]
            query_logits = self.sent_head(query_repr)

            if self.framing.startswith("MC-"):
                cand_repr = self.transformer(batch_inputs["cand_input"])[:, 0]
                cand_logits = self.sent_head(cand_repr)

        elif "-MLM-" in self.framing:
            assert self.framing.startswith("MC-")
            query_repr = self.transformer(batch_inputs["mask_query_input"])
            query_prob = torch.gather(
                F.log_softmax(self.mlm_head(query_repr), dim=2),
                index=batch_inputs["query_input"].unsqueeze(2),
                dim=2,
            )
            query_mask = (batch_inputs["mask_query_input"] == self.mask_id).float().unsqueeze(dim=2)
            query_logits = torch.sum(query_prob * query_mask, dim=1) / query_mask.sum(dim=1)

            cand_repr = self.transformer(batch_inputs["mask_cand_input"])
            cand_prob = torch.gather(
                F.log_softmax(self.mlm_head(cand_repr), dim=2),
                index=batch_inputs["cand_input"].unsqueeze(2),
                dim=2,
            )
            cand_mask = (batch_inputs["mask_cand_input"] == self.mask_id).float().unsqueeze(dim=2)
            cand_logits = torch.sum(cand_prob * cand_mask, dim=1) / cand_mask.sum(dim=1)
        # TODO: check mlm loss in fairseq

        # query_logits: (bs,)
        if self.framing.startswith("MC-"):
            # cand_logits: (batch_cand_count,)
            padding_logits = torch.ones_like(cand_logits[0:1]) * self.padding_logits
            extend_cand_logits = torch.cat([padding_logits, cand_logits])
            extend_cand_matching_idx = batch_inputs["cand_matching_idx"] + 1
            cand_logits = extend_cand_logits[extend_cand_matching_idx.flatten()].view_as(
                extend_cand_matching_idx
            )
            # cand_logits: (bs, max_cands)

        label = batch_inputs["label"]
        if self.framing in ["P-SPAN", "P-SENT", "MC-SENT-PLOSS"]:
            loss = F.binary_cross_entropy_with_logits(query_logits, label, reduction="mean")
        elif self.framing in ["MC-SENT-NOSCALE", "MC-SENT-NOPAIR", "MC-SENT"]:

        batch_outputs = {"loss": loss, "label_pred": label_pred, "acc": acc}
