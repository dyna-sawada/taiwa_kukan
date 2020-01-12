
import glob
import random

def main():
    topics = [t.split("/")[-1] for t in glob.glob("./topic/*")]
    random.seed(1985)
    random.shuffle(topics)

    for test_topic_i in range(len(topics)):
        val_topic_i = test_topic_i-1 if test_topic_i > 0 else len(topics)-1
        train_topic_i = [i for i in range(len(topics)) if i != test_topic_i and i != val_topic_i]

        print("""
python train.py --trial 0 \\
    --train-topic {} --val-topic {} --test-topic {} -out models/outdomain/roberta_lr1e-6_ft/{} \\
    -lr 1e-6 --grad-accum 8 --batch-size 1 --epoch 10 --encoder-finetune
""".format(
            ",".join([topics[i] for i in train_topic_i]),
            topics[val_topic_i],
            topics[test_topic_i],
            topics[test_topic_i],
        ))

if __name__ == "__main__":
    main()