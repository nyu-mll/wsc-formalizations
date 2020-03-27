import os
import pickle
from fairseq_wsc.wsc_utils import (
    wsc_jsonl_iterator,
    winogrande_jsonl_iterator,
    filter_noun_chunks,
    extended_noun_chunks,
)


class WSCTypeTask(object):
    def __init__(self, framing, dataset, reload_data, data_dir, cache_dir):
        self.framing = framing
        self.dataset = dataset
        self.reload_data = reload_data
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.raw_data = None
        self.preprocessed_data = None
        self.iterators = None

        cache_file = os.path.join(self.cache_dir, f"{self.dataset}.pkl")
        if os.path.exist(cache_file):
            with open(cache_file, "rb") as f:
                self.raw_data = pickle.load(f)
        else:
            if self.dataset.startwith("winogrande"):
                self.load_winogrande_data()
            elif self.dataset == "wsc":
                self.load_wsc_data()
            with open(cache_file, "wb") as f:
                pickle.dump(self.raw_data, f)

    def load_wsc_data(self):
        def load_wsc_split(filename, split):
            data_pack = {
                "prefix": [],
                "suffix": [],
                "leading_space": [],
                "trailing_space": [],
                "query": [],
                "candidate": [],
                "p_label": [],
                "mc_label": [],
            }

            record = {}
            sentences = []

            split_file = os.path.join(self.data_dir, "WSC", filename)
            for sentence, pronoun_span, query, label in wsc_jsonl_iterator(split_file):
                data_pack["prefix"].append(sentence[: pronoun_span.start].text)
                data_pack["suffix"].append(sentence[pronoun_span.end :].text_with_ws)
                # spaCy spans include trailing spaces, but we need to know about
                # leading spaces for the GPT-2 BPE
                data_pack["leading_space"].append(
                    " " if sentence[: pronoun_span.start].text_with_ws.endswith(" ") else ""
                )
                data_pack["trailing_space"].append(
                    " " if pronoun_span.text_with_ws.endswith(" ") else ""
                )
                # get noun phrases, excluding pronouns and anything overlapping with the query
                cand_spans = filter_noun_chunks(
                    extended_noun_chunks(sentence),
                    exclude_pronouns=True,
                    exclude_query=query,
                    exact_match=False,
                )

                data_pack["query"].append(query)
                data_pack["candidate"].append([cand_span.text for cand_span in cand_spans])
                data_pack["p_label"].append(int(label))
                if split != "test":
                    sentences.append(sentence.text)
                    if label:
                        record[sentence.text] == query
                else:
                    data_pack["mc_label"].append(0)

            for i, sentence in enumerate(sentences):
                if sentence in record:
                    correct = record["sentence"]
                    data_pack["mc_label"].append(
                        ([data_pack["query"][i]] + data_pack["candidate"][i]).index(correct)
                    )
                else:
                    data_pack["mc_label"].append(-1)

            return data_pack

        self.raw_data = {
            "train": load_wsc_split("train.jsonl", "train"),
            "val": load_wsc_split("val.jsonl", "val"),
            "test": load_wsc_split("test.jsonl", "test"),
        }

    def load_winogrande_data(self):
        def load_winogrande_split(filename, split):
            data_pack = {
                "prefix": [],
                "suffix": [],
                "leading_space": [],
                "trailing_space": [],
                "query": [],
                "candidate": [],
                "p_label": [],
                "mc_label": [],
            }

            split_file = os.path.join(self.data_dir, "Winogrande", filename)
            for sentence, pronoun_span, query, cand_text in winogrande_jsonl_iterator(split_file):
                data_pack["prefix"].append(sentence[: pronoun_span[0]].rstrip())
                data_pack["suffix"].append(sentence[pronoun_span[1] :])
                data_pack["leading_space"].append(
                    " " if sentence[: pronoun_span[0]].endswith(" ") else ""
                )
                data_pack["trailing_space"].append("")
                data_pack["query"].append(query)
                data_pack["candidate"].append([cand_text])
                data_pack["p_label"].append(1)
                data_pack["mc_label"].append(0)
                if split != "test":
                    data_pack["prefix"].append(sentence[: pronoun_span[0]].rstrip())
                    data_pack["suffix"].append(sentence[pronoun_span[1] :])
                    data_pack["leading_space"].append(
                        " " if sentence[: pronoun_span[0]].endswith(" ") else ""
                    )
                    data_pack["trailing_space"].append("")
                    data_pack["query"].append(cand_text)
                    data_pack["candidate"].append([query])
                    data_pack["p_label"].append(0)
                    data_pack["mc_label"].append(1)
            return data_pack

        training_size = self.dataset.split("_")[-1]
        self.raw_data = {
            "train": load_winogrande_split(f"train_{training_size}", "train"),
            "val": load_winogrande_split("dev.jsonl", "val"),
            "test": load_winogrande_split("test.jsonl", "test"),
        }

    def preprocess_data(self, model):
        raise NotImplementedError

    def build_iterators(self, bs):
        raise NotImplementedError

    def write_pred(self, pred):
        raise NotImplementedError
