from fairseq_wsc.wsc_utils import wsc_jsonl_iterator
from fairseq_wsc.wsc_utils import winogrande_jsonl_iterator


class WSCTypeTask(object):
    def __init__(self, data_dir, exp_dir, dataset, framing):
        self.data_dir = data_dir
        self.exp_dir = exp_dir
        self.dataset = dataset
        self.framing = framing
        self.raw_data = None
        self.preprocessed_data = None
        self.iterators = None

        if self.dataset == "winogrande":
            self.load_winogrande_data()
        elif self.dataset == "wsc":
            self.load_wsc_data()
        self.preprocess_data()
        self.build_iterators()

    def load_winogrande_data(self):
        raise NotImplementedError

    def load_wsc_data(self):
        raise NotImplementedError

    def preprocess_data(self):
        raise NotImplementedError

    def build_iterators(self):
        raise NotImplementedError


if __name__ == "__main__":
    wsc_train = "/scratch/hl3236/data/WSC/train.jsonl"
    wsc_val = "/scratch/hl3236/data/WSC/val.jsonl"
    winogrande_train = "/scratch/hl3236/data/Winogrande/train_xl.jsonl"
    winogrande_val = "/scratch/hl3236/data/Winogrande/dev.jsonl"
    wsc_data = list(wsc_jsonl_iterator(wsc_val))
    winogrande_data = list(winogrande_jsonl_iterator(winogrande_val))
    import IPython

    IPython.embed()
