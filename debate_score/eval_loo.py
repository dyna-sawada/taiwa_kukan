
import glob
import random

from data import DebateSet

def main():
    # Load all debates.
    dataset = {t.split("/")[-1]: DebateSet.from_dir(t) for t in glob.glob("./topic/*")}
    whole_dbs = DebateSet.concat(dataset.values())
    whole_dbs.to_json("./all_debates.json")

    print(len(whole_dbs.speeches), "instances found.")

if __name__ == "__main__":
    main()