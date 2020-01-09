
import logging

import torch
import set_debates
import set_scores

class DebateSet:
    def __init__(self, debates, speeches, scores):
        self.debates = debates
        self.speeches = speeches
        self.scores = scores

    @staticmethod
    def from_dir(topic):
        debate_lists = set_debates.set_speech_list(
            "{}/debates.txt".format(topic),
            "{}/orders.txt".format(topic),
        )

        speeches_list = set_debates.flatten(debate_lists)

        with open("{}/preprocessed_speech_list.txt".format(topic), "w") as f:
            for s in speeches_list:
                print(s, file=f)

        score_list = set_scores.set_score_list(
            "{}/scores.txt".format(topic),
            "{}/orders.txt".format(topic),
        )
        score_list = set_scores.flatten(score_list)

        return DebateSet(debate_lists, speeches_list, score_list)

    @staticmethod
    def concat(debatesets: list):
        debates, speeches, scores = [], [], []

        for dbs in debatesets:
            debates.extend(dbs.debates)
            speeches.extend(dbs.speeches)
            scores.extend(dbs.scores)

        return DebateSet(debates, speeches, scores)

    def to_model_input(self, tokenizer):
        # token
        speech_tokens = [tokenizer.tokenize(t)[-511:] for t in self.speeches]
        speech_ids = [tokenizer.encode(t,
                                       add_special_tokens=True,
                                       max_length=512,
                                       pad_to_max_length=512) for t in speech_tokens]

        logging.info(tokenizer.decode(speech_ids[0], clean_up_tokenization_spaces=False))
        logging.info(tokenizer.decode(speech_ids[1], clean_up_tokenization_spaces=False))

        return torch.utils.data.dataset.TensorDataset(
            torch.tensor(speech_ids),
            torch.tensor(self.scores),
        )
