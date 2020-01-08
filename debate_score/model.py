
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

    def get_tokenizer(self):
        return self.tok

    def fit(self,
            xy_train: torch.utils.data.dataset.TensorDataset,
            xy_val: torch.utils.data.dataset.TensorDataset,
            fn_save_to,
            ):
        optimizer = optim.Adam(self.scorer.parameters(), lr=self.args.learning_rate)
        best_val_mse = 9999

        print("Start training...")

        for epoch in range(self.args.epochs):
            print("Epoch", epoch)

            train_loss = self.fit_epoch(xy_train, optimizer)

            with torch.no_grad():
                val_loss = self.validate(xy_val)

            print("MSE", "Train:", train_loss, "Val:", val_loss)
            print("RMSE", "Train:", np.sqrt(train_loss), "Val:", np.sqrt(val_loss))

            if best_val_mse > val_loss:
                print("Best validation loss. Saving to {}...".format(fn_save_to))
                torch.save(self.scorer.state_dict(), fn_save_to)

                best_val_mse = val_loss


    def fit_epoch(self, xy_train: torch.utils.data.dataset.TensorDataset, optimizer):
        running_loss = 0.0

        self.scorer.train()
        self.scorer.roberta.eval()

        iter_train = torch.utils.data.DataLoader(
            xy_train,
            batch_size=self.args.batch_size,
            shuffle=True,
        )

        for batch in tqdm(iter_train):
            speeches, y_true = (d.to(self.device) for d in batch)

            optimizer.zero_grad()

            # Forward pass
            y_pred = self.scorer(speeches)
            loss = self.loss_fn(y_pred, y_true)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / len(xy_train)

    def validate(self, xy_val: torch.utils.data.dataset.TensorDataset):
        running_loss = 0.0

        self.scorer.eval()
        self.scorer.roberta.eval()

        iter_val = torch.utils.data.DataLoader(
            xy_val,
            batch_size=self.args.batch_size,
        )

        for batch in tqdm(iter_val):
            speeches, y_true = (d.to(self.device) for d in batch)

            y_pred = self.scorer(speeches)
            loss = self.loss_fn(y_pred, y_true)

            running_loss += loss.item()

        return running_loss / len(xy_val)