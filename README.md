# フォルダ、ファイル説明

- text : 各試合の文字書き起こしwordファイルが入っています
- topic : 論題ごとにファイルわけをしています　中身は(1.debates.txt：ディベートテキスト　2.orders.txt：スピーチの順番　3.scores.txt：点数)
    - scores.txt：一つの試合につきスコアの数は11に揃えてあります。set_scores.pyでorders.txtと組み合わせて[[ディベート１のスコア], [ディベート2のスコア], ...]に変形できるようにしてあります。
    - debates.txt：一つの試合につきスピーチの数は14に揃えてあります。set_debates.pyとorders.txtと組み合わせて変形できるようにしてあります。
- roberta_linear_test.ipynb：メインコード（論題が同じ５試合すべてを訓練にまわしたときのLossの値がこれ）
- set_debates.py：スピーチ前処理の関数（まとまっているディベートをそれぞれ[立論、立論＋反論、...]という形に変換）
- set_scores.py：スコア前処理の関数（ディベートごとに二重リスト、ないしフラットにかえれる）

# How to run

## Training
```
CUDA_VISIBLE_DEVICES=0 python train.py -out models/microchip
```

## Evaluation

TBA