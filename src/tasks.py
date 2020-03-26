import os
import pickle
from fairseq_wsc.wsc_utils import wsc_jsonl_iterator
from fairseq_wsc.wsc_utils import winogrande_jsonl_iterator


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
            split_file = os.path.join(self.data_dir, "WSC", filename)
            for sentence, pronoun_span, query, label in wsc_jsonl_iterator(split_file):
                pass

        self.raw_data = {
            "train": load_wsc_split("train.jsonl", "train"),
            "val": load_wsc_split("val.jsonl", "val"),
            "test": load_wsc_split("test.jsonl", "test"),
        }

    def load_winogrande_data(self):
        def load_winogrande_split(filename, split):
            split_file = os.path.join(self.data_dir, "Winogrande", filename)
            for sentence, pronoun_span, query, cand_text in winogrande_jsonl_iterator(split_file):
                pass

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
