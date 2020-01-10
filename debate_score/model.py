
import json
import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score

from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


class DummyLR:
    def __init__(self):
        pass

    def step(self):
        pass


class TorchDebateScorer(nn.Module):
    def __init__(self, args):
        super(TorchDebateScorer, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-large",
                                                    output_hidden_states=True)
        self.fc1 = nn.Linear(1024, 1024)
        self.args = args

        if args.binary_class:
            self.fc2 = nn.Linear(1024, 2)

        else:
            self.fc2 = nn.Linear(1024, 1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        _, _, hidden_states = self.roberta(input_ids=x)
        # out = torch.mean(hidden_states[-3], 1)
        out = hidden_states[-1][:, 0]

        if self.args.binary_class:
            out = self.fc1(out)
            out = self.tanh(out)
            out = self.fc2(out)

        else:
            out = self.fc2(out)
            out = self.tanh(out) * 50
            out = out[:, 0]

        return out


class DebateScorer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.scorer = TorchDebateScorer(self.args).to(self.device)
        self.tok = RobertaTokenizer.from_pretrained("roberta-large")

        if args.binary_class:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        else:
            self.loss_fn = nn.MSELoss()

    @staticmethod
    def from_pretrained(fn_model_dir, args, device):
        fn_model = os.path.join(fn_model_dir, "best_model.pt")
        logging.info("Loading model from {}...".format(fn_model))

        m = DebateScorer(args, device)
        m.scorer.load_state_dict(torch.load(fn_model))
        m.scorer = m.scorer.to(device)

        return m

    def get_tokenizer(self):
        return self.tok

    def fit(self,
            xy_train: torch.utils.data.dataset.TensorDataset,
            xy_val: torch.utils.data.dataset.TensorDataset,
            fn_save_to_dir,
            ):
        self.scorer.train()

        if self.args.encoder_finetune:
            self.scorer.roberta.train()
        else:
            self.scorer.roberta.eval()

        if self.args.encoder_finetune:
            no_decay = ['bias', 'LayerNorm.weight']
            trainable_params = [
                {'params': [p for n, p in self.scorer.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in self.scorer.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            t_total = ((len(xy_train) + self.args.batch_size - 1) // self.args.batch_size) * self.args.epochs
            optimizer = AdamW(trainable_params, lr=self.args.learning_rate, eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=int(t_total * 0.1),
                                                        num_training_steps=t_total)

        else:
            trainable_params = list(self.scorer.fc1.parameters()) + list(self.scorer.fc2.parameters())
            optimizer = optim.Adam(trainable_params,
                                   lr=self.args.learning_rate)
            scheduler = DummyLR()

        best_val_mse = 9999
        train_losses, val_losses = [], []

        logging.info("Start training...")
        logging.info("Trainable parameters: {}".format(len(trainable_params)))

        for epoch in range(self.args.epochs):
            logging.info("Epoch {}".format(1+epoch))

            train_loss = self.fit_epoch(xy_train, optimizer, scheduler)

            with torch.no_grad():
                val_loss, y_val_pred, y_val_true = self.validate(xy_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logging.info("Loss Train: {} Val: {}".format(train_loss, val_loss))

            if self.args.binary_class:
                logging.info("Acc. Val: {}".format(accuracy_score(y_val_true, y_val_pred.argmax(axis=1))))

            else:
                logging.info("RMSE Train: {} Val: {}".format(np.sqrt(train_loss), np.sqrt(val_loss)))

            if best_val_mse > val_loss:
                logging.info("Best validation loss. Saving to {}...".format(fn_save_to_dir))
                torch.save(self.scorer.state_dict(), os.path.join(fn_save_to_dir, "best_model.pt"))

                best_val_mse = val_loss

        with open(os.path.join(fn_save_to_dir, "train_log.json"), "w") as f:
            train_log = {
                "train_losses": train_losses,
                "val_losses": val_losses,
            }

            json.dump(train_log, f)

    def fit_epoch(self, xy_train: torch.utils.data.dataset.TensorDataset, optimizer, scheduler):
        running_loss = []
        y_preds, y_trues = [], []
        grad_accum_steps = 0

        self.scorer.train()

        if self.args.encoder_finetune:
            self.scorer.roberta.train()
        else:
            self.scorer.roberta.eval()

        iter_train = torch.utils.data.DataLoader(
            xy_train,
            batch_size=self.args.batch_size,
            shuffle=True,
        )

        optimizer.zero_grad()

        for batch in tqdm(iter_train):
            speeches, y_true = (d.to(self.device) for d in batch)

            # Forward pass
            y_pred = self.scorer(speeches)
            loss = self.loss_fn(y_pred, y_true) / self.args.grad_accum

            y_preds.extend(y_pred.cpu().detach().numpy())
            y_trues.extend(y_true.cpu().detach().numpy())

            running_loss += [loss.item()]
            grad_accum_steps += 1

            # Backward pass
            loss.backward()

            if grad_accum_steps % self.args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        y_preds, y_trues = np.array(y_preds), np.array(y_trues)

        if self.args.binary_class:
            y_preds = y_preds.argmax(axis=1)

        logging.info("Prediction sample: {}".format(y_preds))
        logging.info("Target sample: {}".format(y_trues))

        if self.args.binary_class:
            logging.info("Acc. Train: {}".format(accuracy_score(y_trues, y_preds)))

        return np.mean(running_loss)

    def validate(self, xy_val: torch.utils.data.dataset.TensorDataset):
        running_loss = []

        self.scorer.eval()
        self.scorer.roberta.eval()

        iter_val = torch.utils.data.DataLoader(
            xy_val,
            batch_size=self.args.batch_size,
        )

        y_preds, y_trues = [], []

        for batch in tqdm(iter_val):
            speeches, y_true = (d.to(self.device) for d in batch)

            y_pred = self.scorer(speeches)
            loss = self.loss_fn(y_pred, y_true)

            y_preds.extend(y_pred.cpu().detach().numpy())
            y_trues.extend(y_true.cpu().detach().numpy())

            running_loss += [loss.item()]

        return np.mean(running_loss), np.array(y_preds), np.array(y_trues)

    def test(self, xy_test, fn_save_to_dir):
        logging.info("Start evaluation...")

        with torch.no_grad():
            val_loss, y_pred, y_true = self.validate(xy_test)

        result_log = {
            "prediction": y_pred.tolist(),
            "gold": y_true.tolist(),
        }

        if self.args.binary_class:
            result_log["class"] = y_pred.argmax(axis=1)
            result_log["acc"] = accuracy_score(y_true, y_pred.argmax(axis=1))

        else:
            result_log["rmse"] = np.sqrt(val_loss)

        print(result_log)

        logging.info("Results are stored in {}/results.json.".format(fn_save_to_dir))

        with open(os.path.join(fn_save_to_dir, "results.json"), "w") as f:
            json.dump(result_log, f)
