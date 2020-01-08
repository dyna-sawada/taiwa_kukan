
import torch
import set_debates

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

        with open("{}/scores.txt".format(topic), 'r') as h:
            scores = h.read()
            score_list = [float(s) for s in scores.split()]

        return DebateSet(debate_lists, speeches_list, score_list)

    def to_model_input(self, tokenizer):
        # token
        speech_tokens = list(
            map(
                lambda t: tokenizer.tokenize(t)[-511:], self.speeches
            )
        )

        # encode-ids
        speech_ids = list(
            map(
                lambda t: tokenizer.encode(t, add_special_tokens=True), speech_tokens
            )
        )

        # 要素の大きさを512につめる
        pad_tok = tokenizer.encoder.get(tokenizer.pad_token)
        speech_ids = [
            ids[:512] + [pad_tok]*(512-len(ids)) for ids in speech_ids
        ]

        return torch.utils.data.dataset.TensorDataset(
            torch.tensor(speech_ids),
            torch.tensor(self.scores[:len(speech_ids)]),
        )
