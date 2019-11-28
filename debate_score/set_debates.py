# test

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

def make_speech_list(Debate, Order):
    # 新たなリストに追加
    # order = [G1立論,G1反論,G1再構築,G1再反論,G2立論,G2反論,O1立論,O1反論,O1再構築,O2立論,O2反論]

    speech_list = []
    if Order[0] != 0:
         speech_list.append(Debate[Order[0]])
    if Order[0] != 0 and Order[1] != 0:
        speech_list.append(Debate[Order[0]]+" "+Debate[Order[1]])
    if Order[0] != 0 and Order[1] != 0 and Order[2] != 0:
        speech_list.append(Debate[Order[0]]+" "+Debate[Order[1]]+" "+Debate[Order[2]])
    if Order[0] != 0 and Order[1] != 0 and Order[2] != 0 and Order[3] != 0:
        speech_list.append(Debate[Order[0]]+" "+Debate[Order[1]]+" "+Debate[Order[2]]+" "+Debate[Order[3]])
    if Order[4] != 0:
        speech_list.append(Debate[Order[4]])
    if Order[4] != 0 and Order[5] != 0:
        speech_list.append(Debate[Order[4]]+" "+Debate[Order[5]])
    if Order[6] != 0:
        speech_list.append(Debate[Order[6]])
    if Order[6] != 0 and Order[7] != 0:
        speech_list.append(Debate[Order[6]]+" "+Debate[Order[7]])
    if Order[6] != 0 and Order[7] != 0 and Order[8] != 0:
        speech_list.append(Debate[Order[6]]+" "+Debate[Order[7]]+" "+Debate[Order[8]])
    if Order[9] != 0:
        speech_list.append(Debate[Order[9]])
    if Order[9] != 0 and Order[10] != 0:
        speech_list.append(Debate[Order[9]]+" "+Debate[Order[10]])

    return speech_list   #返り：変換されたリスト

def set_speech_list(D, O):
    # ディベート（１４に全部揃える）を[[ディベート1],[ディベート2],...,[ディベートN]]の形に
    with open(D, 'r') as f:
        speeches = f.read()
        speeches_kai = re.sub("\[applause\]", "", speeches)
        split_speeches_list = speeches_kai.split("\n")
        debate_lists = list(split_list(split_speeches_list, 14))

    # スピーチ順番（全部11固定）を[[ディベート1],[ディベート2],...,[ディベートN]]の形に
    # orders を　文字から数字に変更
    with open(O, 'r') as g:
        orders = g.read()
        orders_list = orders.split()
        orders_list = [int(s) for s in orders_list]
        order_lists = list(split_list(orders_list, 11))

    speeches_lists = []
    for (debate_list, order_list) in zip(debate_lists, order_lists):
        speeches_lists.append(make_speech_list(debate_list, order_list))
    return speeches_lists


"""
# 書き出しチェック
str_ = '\n'.join(speeches_list)
with open("result_test.txt", 'w') as f:
    f.write(str_)
    """
