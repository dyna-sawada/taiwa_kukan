import glob
import json

import numpy as np


def main():
    all_rmse = {}

    for fn_ret in glob.glob("models/*/results.json"):
        ret = json.load(open(fn_ret))

        all_rmse[fn_ret.split("/")[1]] = ret["rmse"]

    print("# Cross-topic leave-one-out")
    print(len(all_rmse), "results are available.")
    print("RMSE: {:.1f}".format(np.mean(list(all_rmse.values()))))
    print("Details:")

    for topic in all_rmse:
        print(" ", topic, all_rmse[topic])

    print()
    print("# In-topic leave-one-out")
    print("RMSE:", "Not implemented yet")

if __name__ == "__main__":
    main()