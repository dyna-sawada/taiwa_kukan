## debate_socore

>フォルダ、ファイル説明

text : 各試合の文字書き起こしwordファイルが入っています

topic : 論題ごとにファイルわけをしています　中身は(1.debates.txt：ディベートテキスト　2.orders.txt：スピーチの順番　3.scores.txt：点数)
(注意)夏合宿以外のscores.txtではすべてリストの要素数は11に揃えてあります。set_score.pyでorders.txtと組み合わせて[[ディベート１のスコア], [ディベート2のスコア], ...]に変形できるようにしてあります。
ただし、夏合宿のscores.txtは先にスピーチの個数と揃えています。紛らわしいので、あとで11に揃えます。

roberta_linear_test.ipynb：メインコード（論題が同じ５試合すべてを訓練にまわしたときのLossの値がこれ）

set_debates.py：スピーチ前処理の関数（まとまっているディベートをそれぞれ[立論、立論＋反論、...]という形に変換）
set_scores.py：スコア前処理の関数（ディベートごとに二重リスト、ないしフラットにかえれる）
