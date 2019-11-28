## debate_socore

>フォルダ、ファイル説明

text : 各試合の文字書き起こしwordファイルが入っています

topic : 論題ごとにファイルわけをしています　中身は(1.debates.txt：ディベートテキスト　2.orders.txt：スピーチの順番　3.scores.txt：点数)

roberta_linear_test.ipynb：メインコード（論題が同じ５試合すべてを訓練にまわしたときのLossの値がこれ）

set_debates.py：スピーチ前処理の関数（まとまっているディベートをそれぞれ[立論、立論＋反論、...]という形に変換）

