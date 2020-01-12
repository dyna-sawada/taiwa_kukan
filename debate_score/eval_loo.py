
import glob
import random
import collections

from data import DebateSet

def main():
    # Load all debates.
    dataset = {t.split("/")[-1]: DebateSet.from_dir(t) for t in glob.glob("./topic/*")}

    for topic in dataset:
        if len(dataset[topic].table_keys) < 3:
            continue

        for table_id_test in dataset[topic].table_keys:
            print("""
python train.py --trial 0 \\
    --loo-test-target {}:{} -out models/indomain/roberta_lr1e-6_ft/{}_{} \\
    -lr 1e-6 --grad-accum 8 --batch-size 1 --epoch 10 --encoder-finetune
""".format(
                topic,
                table_id_test,
                topic,
                table_id_test,
            ))

if __name__ == "__main__":
    main()