import argparse

import glob
import json
import os

import numpy as np


def main(args):
    all_rmse = {}

    for fn_ret in glob.glob(os.path.join(args.target_model, "*/results.json")):
        ret = json.load(open(fn_ret))

        all_rmse[fn_ret.split("/")[-2]] = ret["rmse"]

    print("# Cross-topic leave-one-out")
    print(len(all_rmse), "results are available.")
    print("--")
    for topic in all_rmse:
        print("{}\t{:.1f}".format(topic, all_rmse[topic]))

    print("--")
    print("Overall\t{:.1f} (±{:.1f})".format(
        np.mean(list(all_rmse.values())),
        np.std(list(all_rmse.values())),
    ))


    print()
    print("# In-topic leave-one-out")
    print("RMSE:", "Not implemented yet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--target-model', required=True,
        help="Model directory to be evaluated.")

    args = parser.parse_args()
    main(args)
