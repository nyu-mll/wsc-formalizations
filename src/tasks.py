import os
import pickle
import logging as log
import json
from fairseq_wsc.wsc_utils import (
    filter_noun_chunks,
    extended_noun_chunks,
    get_detokenizer,
    get_spacy_nlp,
)


def strip_punc(x):
    return x.rstrip('.,;"').lstrip('"')


def shared_length(list_a, list_b):
    for idx in range(min(len(list_a), len(list_b))):
        if list_a[idx] != list_b[idx]:
            return idx
    return idx + 1


def find_first_char_span(text, target, starting_idx=0):
    span = (starting_idx, min(starting_idx + len(target), len(text)))
    found = False
    for idx in range(starting_idx, len(text) - len(target)):
        if text[idx : idx + len(target)].lower() == target:
            span = (idx, idx + len(target))
            found = True
            break
    if not found:
        log.warning(f"span not found: text={text} target={target}")
    return text[slice(*span)], span


def find_likely_char_span(text, target):
    all_spans = []
    pad_text = f" {text} "
    special_chars = """.,;:"' """
    for idx in range(0, len(text) - len(target)):
        if text[idx : idx + len(target)].lower() == target:
            span = (idx, idx + len(target))
            left, right = (pad_text[idx], pad_text[idx + len(target) + 1])
            span_score = int(left in special_chars) + int(right in special_chars)
            all_spans.append((span, span_score))
    all_spans = all_spans.sort(key=lambda x: x[1], reverse=True)
    assert all_spans != []
    if len(all_spans) != 1:
        log.warning(f"more than one span found text={text} target={target}")
    span = all_spans[0]
    return text[slice(*span)], span


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
            elif self.dataset.startwith("wsc"):
                self.load_wsc_data()
            with open(cache_file, "wb") as f:
                pickle.dump(self.raw_data, f)

    def load_wsc_data(self):
        wsc_candidates = self.dataset.split("_")[-1]
        detok = get_detokenizer()
        nlp = get_spacy_nlp()

        def load_wsc_split(filename, split):
            with open(filename, "r") as f:
                examples = [json.loads(line) for line in f.readlines()]

            # single instance process
            # cleanup weird chars, realign spans
            def convert_wsc_example(example):
                new_example = {}
                new_example["uid"] = f'{split}_{example["idx"]}'
                tokens = example["text"].replace("\n", " ").split()
                new_example["text"] = detok.detokenize(tokens)
                new_example["query_text"] = strip_punc(
                    example["target"]["span1_text"].replace("\n", " ").lower()
                )
                new_example["query_text"], new_example["query_char_span"] = find_first_char_span(
                    text=new_example["text"],
                    target=new_example["query_text"],
                    starting_idx=len(detok.detokenize(tokens[: example["target"]["span1_index"]])),
                )
                new_example["pronoun_text"] = strip_punc(
                    example["target"]["span2_text"].replace("\n", " ").lower()
                )
                new_example["pronoun_text"], new_example[
                    "pronoun_char_span"
                ] = find_first_char_span(
                    text=new_example["text"],
                    target=new_example["pronoun_text"],
                    starting_idx=len(detok.detokenize(tokens[: example["target"]["span2_index"]])),
                )
                if wsc_candidates == "spacy":
                    new_example["cand_text_list"] = [
                        cand.text
                        for cand in filter_noun_chunks(
                            extended_noun_chunks(nlp(new_example["text"])),
                            exclude_pronouns=True,
                            exclude_query=new_example["query_text"],
                            exact_match=False,
                        )
                    ]
                else:
                    new_example["cand_text_list"] = []
                if split != "test":
                    new_example["p_label"] = example["label"]
                return new_example

            examples = list(map(convert_wsc_example, examples))

            # cross example process
            global_ans_dict = {}
            for idx, example in enumerate(examples):
                key = (example["text"], example["pronoun_text"])
                if key not in global_ans_dict:
                    global_ans_dict[key] = {"correct_query": None, "idxs": [], "all_cands": []}
                global_ans_dict[key]["idxs"].append(idx)
                global_ans_dict[key]["all_cands"].append(examples["query_text"])
                if example.get(["p_label"], False):
                    global_ans_dict[key]["correct_query"] = example["query_text"]

            for example_group in global_ans_dict.values():
                correct_query = example_group["correct_query"]
                if split == "train":
                    assert correct_query is not None
                for idx in example_group["idxs"]:
                    if wsc_candidates == "cross":
                        examples[idx]["cand_text_list"] = [
                            cand
                            for cand in example_group["all_cands"]
                            if cand != examples[idx]["query_text"]
                        ]
                    if split == "train" and correct_query is not None:
                        query_and_cands = [examples[idx]["query_text"]] + examples[idx][
                            "cand_text_list"
                        ]
                        try:
                            examples[idx]["mc_label"] = query_and_cands.index(correct_query)
                        except ValueError:
                            examples[idx]["cand_text_list"].insert(0, correct_query)
                            examples[idx]["mc_label"] = 1
            return examples

        self.raw_data = {
            "train": load_wsc_split("train.jsonl", "train"),
            "val": load_wsc_split("val.jsonl", "val"),
            "test": load_wsc_split("test.jsonl", "test"),
        }

    def load_winogrande_data(self):
        detok = get_detokenizer()

        def load_winogrande_split(filename, split):
            with open(filename, "r") as f:
                examples = [json.loads(line) for line in f.readlines()]

            def convert_winogrande_example(example, flip=False):
                new_example = {}
                new_example["uid"] = f'{split}_{example["idx"]}{"f" if flip else ""}'
                new_example["text"] = detok.detokenize(example["sentence"].split())
                new_example["query_text"] = example["option2"] if flip else example["option1"]
                new_example["query_char_span"] = find_likely_char_span(
                    new_example["text"], new_example["query_text"]
                )
                new_example["pronoun_text"] = "_"
                new_example["pronoun_char_span"] = find_likely_char_span(new_example["text"], "_")
                new_example["cand_text_list"] = [example["option1"] if flip else example["option2"]]
                if split != "test":
                    new_example["p_label"] = example["answer"] == ("2" if flip else "1")
                    new_example["mc_label"] = example["answer"] == ("1" if flip else "2")

            if split == "train":
                examples = [convert_winogrande_example(example) for example in examples] + [
                    convert_winogrande_example(example, flip=True) for example in examples
                ]
            else:
                examples = [convert_winogrande_example(example) for example in examples]

        training_size = self.dataset.split("_")[-1]
        self.raw_data = {
            "train": load_winogrande_split(f"train_{training_size}", "train"),
            "val": load_winogrande_split("dev.jsonl", "val"),
            "test": load_winogrande_split("test.jsonl", "test"),
        }

    def preprocess_data(self, model):
        # TODO
        # add tokenize and aligned to tokenized span
        # remove instances accordingly
        # pad the input
        # convert to tensors
        raise NotImplementedError

    def build_iterators(self, bs):
        raise NotImplementedError

    def write_pred(self, pred):
        raise NotImplementedError
