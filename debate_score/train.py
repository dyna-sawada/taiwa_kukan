import argparse
import glob
import os
import random

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
    print(device)

    # Prepare the model and train!
    m = DebateScorer(args, device)
    tokenizer = m.get_tokenizer()

    os.system("mkdir -p {}".format(args.model_dir))

    m.fit(dataset["microchip"].to_model_input(tokenizer),
          dataset["part-time-job"].to_model_input(tokenizer),
          os.path.join(args.model_dir, "best_model.pt"),
          )


if __name__ == "__main__":
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
        '-pytr', '--pytrcache-path', default="/work01/naoya-i/pytr",
        help="Path to pytorch-transformers cache dir.")
    args = parser.parse_args()

    main(args)