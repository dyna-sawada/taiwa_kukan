
import json
import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm


class TorchDebateScorer(nn.Module):
    def __init__(self, args):
        super(TorchDebateScorer, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.fc1 = nn.Linear(1024, 1)

    def forward(self, x):
        last_hidden_states, _ = self.roberta(input_ids=x)
        out = self.fc1(last_hidden_states[:, 0])
        out = out[:, 0]
        return out


class DebateScorer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.scorer = TorchDebateScorer(self.args).to(self.device)
        self.tok = RobertaTokenizer.from_pretrained("roberta-large")

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
        optimizer = optim.Adam(self.scorer.parameters(), lr=self.args.learning_rate)
        best_val_mse = 9999
        train_losses, val_losses = [], []

        logging.info("Start training...")

        for epoch in range(self.args.epochs):
            logging.info("Epoch {}".format(epoch))

            train_loss = self.fit_epoch(xy_train, optimizer)

            with torch.no_grad():
                val_loss, _ = self.validate(xy_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logging.info("MSE Train: {} Val: {}".format(train_loss, val_loss))
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

    def fit_epoch(self, xy_train: torch.utils.data.dataset.TensorDataset, optimizer):
        running_loss = []
        grad_accum_steps = 0

        self.scorer.train()
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

            running_loss += [loss.item()]
            grad_accum_steps += 1

            # Backward pass
            loss.backward()

            if grad_accum_steps % self.args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        return np.mean(running_loss)

    def validate(self, xy_val: torch.utils.data.dataset.TensorDataset):
        running_loss = []

        self.scorer.eval()
        self.scorer.roberta.eval()

        iter_val = torch.utils.data.DataLoader(
            xy_val,
            batch_size=self.args.batch_size,
        )

        y_preds = []

        for batch in tqdm(iter_val):
            speeches, y_true = (d.to(self.device) for d in batch)

            y_pred = self.scorer(speeches)
            loss = self.loss_fn(y_pred, y_true)

            y_preds.extend(y_pred.cpu().detach().numpy())

            running_loss += [loss.item()]

        return np.mean(running_loss), np.array(y_preds)

    def test(self, xy_test, fn_save_to_dir):
        logging.info("Start evaluation...")

        with torch.no_grad():
            val_loss, y_pred = self.validate(xy_test)

        result_log = {
            "rmse": np.sqrt(val_loss),
            "prediction": y_pred.tolist()
        }

        print(result_log)

        logging.info("Results are stored in {}/results.json.".format(fn_save_to_dir))

        with open(os.path.join(fn_save_to_dir, "results.json"), "w") as f:
            json.dump(result_log, f)
