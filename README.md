# フォルダ、ファイル説明

- text : 各試合の文字書き起こしwordファイルが入っています
- topic : 論題ごとにファイルわけをしています　中身は(1.debates.txt：ディベートテキスト　2.orders.txt：スピーチの順番　3.scores.txt：点数)
    - scores.txt：一つの試合につきスコアの数は11に揃えてあります。set_scores.pyでorders.txtと組み合わせて[[ディベート１のスコア], [ディベート2のスコア], ...]に変形できるようにしてあります。
    - debates.txt：一つの試合につきスピーチの数は14に揃えてあります。set_debates.pyとorders.txtと組み合わせて変形できるようにしてあります。
    - orders.txt:[G1立論, G1反論, G1再構築, G1再反論, G2立論, G2反論, O1立論, O1反論, O1再構築, O2立論, O2反論]の形にスピーチの順番を並べたものになります。　話されなかったスピーチの番号はここでは0としてあります。
    
- roberta_linear_test.ipynb：メインコード（論題が同じ５試合すべてを訓練にまわしたときのLossの値がこれ）
- set_debates.py：スピーチ前処理の関数（まとまっているディベートをそれぞれ[立論、立論＋反論、...]という形に変換）
- set_scores.py：スコア前処理の関数（ディベートごとに二重リスト、ないしフラットにかえれる）

# How to run

## Training & Evaluation

Run the following command:

```
CUDA_VISIBLE_DEVICES=0 python train.py --trial 0 -lr 1e-3 \
    --train-topic four-day-work --val-topic part-time-job --test-topic microchip \
    --model-dir models/microchip
```

You can specify topics for training, validation and testing by `--*-topic` parameters.

This yields the following files in `models/microchip`:

- `best_model.pt`: the best model (on validation set)
- `results.json`: RMSE and prediction on testset.
- `train_log.json`: training log containing train loss and validation loss.


## Automated evaluation
### Leave-one-out evaluation (in-domain testing)

```
python eval_loo.py | CUDA_VISIBLE_DEVICES=0 zsh
```

### Cross-topic leave-one-out evaluation (out-domain testing)

```
python eval_topic_loo.py | CUDA_VISIBLE_DEVICES=0 zsh
```

## Summarizing results

```
python summarize_results.py
```
