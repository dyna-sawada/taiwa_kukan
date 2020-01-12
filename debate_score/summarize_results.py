import argparse
import collections

import glob
import json
import os

import numpy as np


def main(args):
    all_rmse = {}

    for fn_ret in glob.glob(os.path.join(args.target_model, "*/results.json")):
        ret = json.load(open(fn_ret))

        all_rmse[fn_ret.split("/")[-2]] = ret["rmse"]

    print(len(all_rmse), "results are available.")
    print("--")

    for topic in all_rmse:
        print("{}\t{:.1f}".format(topic, all_rmse[topic]))

    print("--")

    if args.indomain:
        topic_wise = collections.defaultdict(list)

        for topic in all_rmse:
            topic_wise[topic.split("_")[0]] += [all_rmse[topic]]

        for topic in topic_wise:
            print("{}\t{:.1f} (±{:.1f})".format(topic,
                                                np.mean(topic_wise[topic]),
                                                np.std(topic_wise[topic]),
                                                ))

    print("--")
    print("Overall\t{:.1f} (±{:.1f})".format(
        np.mean(list(all_rmse.values())),
        np.std(list(all_rmse.values())),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--target-model', required=True,
        help="Model directory to be evaluated.")
    parser.add_argument(
        '-i', '--indomain', action="store_true",
        help="Output in-domain specific analysis.")

    args = parser.parse_args()
    main(args)
