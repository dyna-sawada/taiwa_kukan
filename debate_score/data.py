
import logging
import json

import torch
import set_debates
import set_scores

MAX_SEQ_LEN = 512

class DebateSet:
    def __init__(self, speeches, scores, tables):
        self.speeches = speeches
        self.scores = scores
        self.tables = tables

        self.table_keys = list(set(tables))

    @staticmethod
    def from_dir(topic):
        debate_list, table_list = set_debates.set_speech_list(
            "{}/debates.txt".format(topic),
            "{}/orders.txt".format(topic),
        )

        speeches_list = set_debates.flatten(debate_list)

        score_list = set_scores.set_score_list(
            "{}/scores.txt".format(topic),
            "{}/orders.txt".format(topic),
        )
        score_list = set_scores.flatten(score_list)

        with open("{}/preprocessed.txt".format(topic), "w") as f:
            for sp, sc in zip(speeches_list, score_list):
                print(sc, sp, file=f)

        return DebateSet(speeches_list, score_list, table_list)

    @staticmethod
    def concat(debatesets: list):
        speeches, scores, tables = [], [], []

        for dbs in debatesets:
            speeches.extend(dbs.speeches)
            scores.extend(dbs.scores)
            tables.extend(dbs.tables)

        return DebateSet(speeches, scores, tables)

    def filter(self, filter_fn):
        new_dbs = [
            (sp, sc, tb)
            for sp, sc, tb in zip(self.speeches, self.scores, self.tables)
            if filter_fn(sp, sc, tb)
            ]
        new_dbs = list(zip(*new_dbs))

        print(new_dbs)

        return DebateSet(*new_dbs)

    def to_json(self, fn):
        with open(fn, "w") as f:
            json.dump({"speeches": self.speeches,
                       "scores": self.scores, "tables": self.tables}, f)

    def to_model_input(self, args, tokenizer):
        # token
        speech_tokens = [tokenizer.tokenize(t) for t in self.speeches]
        speech_ids = [tokenizer.encode(t,
                                       add_special_tokens=True,
                                       max_length=MAX_SEQ_LEN,
                                       pad_to_max_length=MAX_SEQ_LEN) for t in speech_tokens]

        logging.info(tokenizer.decode(speech_ids[0], clean_up_tokenization_spaces=False))
        logging.info(tokenizer.decode(speech_ids[1], clean_up_tokenization_spaces=False))

        return torch.utils.data.dataset.TensorDataset(
            torch.tensor(speech_ids),
            torch.tensor([int(s>0) if args.binary_class else s for s in self.scores]),
        )
