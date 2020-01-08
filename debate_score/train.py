import argparse
import glob
import os
import random
import json

import logging

import torch
import numpy as np

from data import DebateSet
from model import DebateScorer



def main(args):
    # Fix random seeds.
    i = args.trial
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.backends.cudnn.deterministic = True

    # Load all debates.
    dataset = {t.split("/")[-1]: DebateSet.from_dir(t) for t in glob.glob("./topic/*")}

    # Prepare GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(device)

    train(dataset, args, device)
    test(dataset, args, device)


def train(dataset, args, device):
    # Prepare and train the model!
    m = DebateScorer(args, device)
    tokenizer = m.get_tokenizer()

    os.system("mkdir -p {}".format(args.model_dir))

    with open(os.path.join(args.model_dir, "params.json"), "w") as f:
        json.dump(args.__dict__, f)

    logging.info(args.__dict__)

    m.fit(dataset["microchip"].to_model_input(tokenizer),
          dataset["part-time-job"].to_model_input(tokenizer),
          os.path.join(args.model_dir, "best_model.pt"),
          )


def test(dataset, args, device):
    # Prepare and evaluate the model!
    m = DebateScorer.from_pretrained(os.path.join(args.model_dir, "best_model.pt"),
                                     args, device)
    tokenizer = m.get_tokenizer()

    m.test(dataset["four-day-work"].to_model_input(tokenizer),
           args.model_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s- %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--trial', type=int, required=True,
        help="Trial No.")
    parser.add_argument(
        '-out', '--model-dir', required=True,
        help="Output directory.")

    parser.add_argument(
        '-ep', '--epochs', default=10, type=int,
        help="Max training epochs.")
    parser.add_argument(
        '-bs', '--batch-size', default=2, type=int,
        help="Training batch size.")
    parser.add_argument(
        '-lr', '--learning-rate', default=1e-5, type=float,
        help="Learning rate.")
    parser.add_argument(
        '-ga', '--grad-accum', default=16, type=int,
        help="Gradient accumulation steps.")

    parser.add_argument(
        '-pytr', '--pytrcache-path', default="/work01/naoya-i/pytr",
        help="Path to pytorch-transformers cache dir.")
    args = parser.parse_args()

    main(args)