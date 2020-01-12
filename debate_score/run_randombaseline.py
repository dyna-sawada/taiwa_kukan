import argparse
import glob
import os
import random
import json

import logging

import numpy as np

from data import DebateSet

from sklearn.metrics import mean_squared_error


def main(args):
    # Fix random seeds.
    i = args.trial
    np.random.seed(i)
    random.seed(i)

    # Load all debates.
    dataset = {t.split("/")[-1]: DebateSet.from_dir(t) for t in glob.glob("./topic/*")}

    for topic in dataset:
        os.system("mkdir -p {}".format(os.path.join(args.model_dir, topic)))

        prediction = [random.randint(-5, 5) for t in range(len(dataset[topic].scores))]
        gold = dataset[topic].scores

        log = {
            "rmse": np.sqrt(mean_squared_error(gold, prediction)),
            "prediction": prediction,
            "gold": gold,
        }

        with open(os.path.join(args.model_dir, topic, "results.json"), "w") as f:
            json.dump(log, f)


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

    args = parser.parse_args()

    main(args)