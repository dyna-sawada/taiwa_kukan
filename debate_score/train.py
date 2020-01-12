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

    logging.info("Config: {}".format(json.dumps(args.__dict__, indent=1)))

    # Create training data
    if args.train_topic is not None:
        dbs_train = DebateSet.concat([dataset[t] for t in args.train_topic.split(",")])
        dbs_val = dataset[args.val_topic]

    if args.loo_test_target is not None:
        topic_test, table_id_test = args.loo_test_target.split(":")
        table_id_test = int(table_id_test)
        table_id_val = table_id_test + 1

        assert(len(dataset[topic_test].table_keys) >= 3)

        if table_id_val not in dataset[topic_test].table_keys:
            table_id_val = 1

        # In-topic construction
        dbs_train = dataset[topic_test].filter(lambda _1, _2, tbl_id: tbl_id != table_id_test and tbl_id != table_id_val)
        dbs_val = dataset[topic_test].filter(lambda _1, _2, tbl_id: tbl_id == table_id_val)

        if not args.indomain_only:
            # Add debates with other topics to the training data
            dbs_train = dbs_train.concat([dataset[t] for t in dataset if t != topic_test])

    logging.info("Training on {} instances.".format(len(dbs_train.speeches)))
    logging.info("Validating on {} instances.".format(len(dbs_val.speeches)))

    m.fit(dbs_train.to_model_input(args, tokenizer),
          dbs_val.to_model_input(args, tokenizer),
          args.model_dir,
          )


def test(dataset, args, device):
    # Prepare and evaluate the model!
    m = DebateScorer.from_pretrained(args.model_dir,
                                     args, device)
    tokenizer = m.get_tokenizer()

    # Create test data
    if args.test_topic is not None:
        dbs_test = dataset[args.test_topic]

    if args.loo_test_target is not None:
        topic, table_id_test = args.loo_test_target.split(":")
        table_id_test = int(table_id_test)

        dbs_test = dataset[topic].filter(lambda _1, _2, tbl_id: tbl_id == table_id_test)

    logging.info("Testing on {} instances.".format(len(dbs_test.speeches)))

    m.test(dbs_test.to_model_input(args, tokenizer),
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

    # Evaluation target
    parser.add_argument(
        '-loo', '--loo-test-target',
        help="Debate to be tested. Format: 'topic:table_id'")
    parser.add_argument(
        '-ind', '--indomain-only', action="store_true",
        help="Use debates with same domain only as training data.")

    parser.add_argument(
        '-trtp', '--train-topic',
        help="Topic to be trained. You can specify multiple topics by colon.")
    parser.add_argument(
        '-vtp', '--val-topic',
        help="Topic to be validated.")
    parser.add_argument(
        '-tstp', '--test-topic',
        help="Topic to be tested.")

    parser.add_argument(
        '-bin', '--binary-class', action="store_true",
        help="Use binary class for prediction.")
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
        '-enc-ft', '--encoder-finetune', action="store_true",
        help="Finetune encoder.")
    parser.add_argument(
        '-enc', '--encoder', default="albert",
        help="Encoder.")

    parser.add_argument(
        '-pytr', '--pytrcache-path', default="/work01/naoya-i/pytr",
        help="Path to pytorch-transformers cache dir.")
    args = parser.parse_args()

    main(args)