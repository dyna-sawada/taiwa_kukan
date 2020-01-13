import argparse
import collections

import glob
import json
import os

import numpy as np
import pandas as pd

import logging


def get_result(args, model, label):
    rows = []
    all_ret = {}
    indomain_mode = "indomain" in model

    for fn_ret in glob.glob(os.path.join(model, "*/results.json")):
        ret = json.load(open(fn_ret))
        folder = fn_ret.split("/")[-2]
        topic = folder.split("_")[0]

        if args.target_topic is not None and \
                topic not in args.target_topic.split(","):
            continue

        all_ret[folder] = ret

    logging.info("{}: {} results are available.".format(model, len(all_ret)))

    for topic in sorted(all_ret):
        if not indomain_mode:
            rows += [collections.OrderedDict(topic=topic,
                                             rmse="{:.1f}".format(all_ret[topic]["rmse"]))]

    if indomain_mode:
        topic_wise = collections.defaultdict(list)

        for topic in all_ret:
            topic_wise[topic.split("_")[0]] += [all_ret[topic]["rmse"]]

        for topic in sorted(topic_wise):
            rows += [collections.OrderedDict(topic=topic,
                                             rmse="{:.1f} (±{:.1f})".format(
                                                np.mean(topic_wise[topic]),
                                                np.std(topic_wise[topic]),
                                                ))]

    all_rmse = [all_ret[t]["rmse"] for t in all_ret]

    rows += [collections.OrderedDict(topic="Overall",
                                     rmse="{:.1f} (±{:.1f})".format(
        np.mean(all_rmse),
        np.std(all_rmse),
    ))]

    df = pd.DataFrame(rows)
    df.columns = ["topic", label]
    df.set_index("topic", inplace=True)

    return df


def main(args):
    dfs = []
    labels = args.labels

    if labels is None:
        labels = args.target_model

    for model, label in zip(args.target_model.split(","), labels.split(",")):
        dfs += [get_result(args, model, label)]

    df = pd.concat(dfs, axis=1, sort=False)

    if args.format == "tsv":
        out = df.to_csv(sep="\t", na_rep="-")
        
    elif args.format == "latex":
        out = df.to_latex(na_rep="-")

    print(out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s- %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--target-model', required=True,
        help="Model directory to be evaluated.")
    parser.add_argument(
        '-l', '--labels',
        help="Labels.")
    parser.add_argument(
        '-tp', '--target-topic',
        help="Topics to be evaluated.")
    parser.add_argument(
        '-f', '--format', default="tsv",
        help="Output format.")

    args = parser.parse_args()
    main(args)
