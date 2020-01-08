#set_score

import math
import re


def split_list(LIST,n):
    # LIST : 全結合してるリスト
    # n : サブリストの要素数
    for idx in range(0, len(LIST), n):
        yield LIST[idx: idx + n]

def flatten(nested_list):
    #2重のリストをフラットにする関数
    return [e for inner_list in nested_list for e in inner_list]

def make_score_list(Score, Order):
    # 新たなリストに追加
    # order = [G1立論,G1反論,G1再構築,G1再反論,G2立論,G2反論,O1立論,O1反論,O1再構築,O2立論,O2反論]
    c = 0
    for i in Order:
        if i == 0:
            Score.pop(c)
            c += 0
        else:
            c += 1
    return Score                # 変換されたスコアリスト


"""
   参考
def make_label_list(label, ga1, ga2, ga3, ga4, gb1, gb2, oa1, oa2, oa3, ob1, ob2):
    label_list = label.split()   # 全部の点数読み込み
    label_list = [float(s) for s in label_list]
    index_list = [ga1, ga2, ga3, ga4, gb1, gb2, oa1, oa2, oa3, ob1, ob2]
    c = 0
    for i in index_list:
        if i == 0:
            label_list.pop(c)
            c += 0
        else:
            c += 1
    return label_list
"""


def set_score_list(S, O):
    # スコア（１１に全部揃える）を[[ディベート1],[ディベート2],...,[ディベートN]]の形に
    with open(S, 'r') as f:
        scores = f.read()
        scores_list = scores.split()
        scores_list = [float(r) for r in scores_list]
        score_lists = list(split_list(scores_list, 11))

    # スピーチ順番（全部11固定）を[[ディベート1],[ディベート2],...,[ディベートN]]の形に
    # orders を　文字から数字に変更
    with open(O, 'r') as g:
        orders = g.read()
        orders_list = orders.split()
        orders_list = [int(s) for s in orders_list]
        order_lists = list(split_list(orders_list, 11))

    scores_lists = []
    for (score_list, order_list) in zip(score_lists, order_lists):
        scores_lists.append(make_score_list(score_list, order_list))
    return scores_lists
