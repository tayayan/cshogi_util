from cshogi import *
from cshogi import KI2
from cshogi.dlshogi import FEATURES1_NUM, FEATURES2_NUM, make_input_features, make_move_label
import sys
import numpy as np
import onnxruntime
import threading
import subprocess
import tkinter as tk
import os

# License: GPLv3 https://www.gnu.org/licenses/gpl-3.0.html

# モデルファイル
modelfile = "model.onnx"
# エンジン
shogiengine = "Suisho5.exe"

#エンジン名・フォント・文字色・背景色
engine = "水匠5"         # エンジン名：水匠
player1 = "先手"
player2 = "後手"
barfont = "BIZ UDGothic"    # フォント：BIZ UDGothic
bgcolor = "#1E2022"         # 背景色：white
fgcolor = "#DDDDDD"         # 文字色：black
fgcolor2 = "#ffd1cc"
fgcolor3 = "#cce6ff"
turnfgcolor = "#f0e9a8"     # 手番の色：#00007f
leftgraphbg = "#aa0000"     # 左グラフの色：#aa0000
rightgraphbg = '#ffff7f'    # 右グラフの色：#ffff7f

def load_options(filepath: str = "info_options.txt") -> None:
    """
    filepath で指定したテキスト（key = value 形式）を読み込み、
    同名のグローバル変数があれば値を文字列として上書きする。
    ・先頭が # の行と空行は無視
    ・値の前後にある余分な空白や引用符 ' " は取り除く
    """
    if not os.path.exists(filepath):
        return                     # ファイルが無ければ既定値のまま

    with open(filepath, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = map(str.strip, line.split("=", 1))

            # 両端に " や ' が付いていれば外す
            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]

            # 既に定義済みのグローバル変数だけを上書き
            if key in globals():
                globals()[key] = val

# 起動時に一度だけ読んで反映
load_options()

# 局面情報
board = Board()

# 推論クラス
player = None

#外部エンジン起動        
shogi = subprocess.Popen(shogiengine, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      encoding="cp932",
                                      errors = "ignore")

class OnnxPolicyPlayer:
    # セッション初期化
    def __init__(self, modelfile:str="model.onnx"):
        print("info string load modelfile {}".format(modelfile))
        # ONNXモデルの推論セッション
        available_providers = onnxruntime.get_available_providers()
        enable_providers = []
        if 'CPUExecutionProvider' in available_providers:
            enable_providers.append('CPUExecutionProvider')
            print("info string enable CPUExecutionProvider")
        self.session = onnxruntime.InferenceSession(modelfile, providers=enable_providers)
        # 入力特徴量の作成
        batchsize_max = 41
        self.features1 = np.empty((batchsize_max, FEATURES1_NUM, 9, 9), dtype=np.float32)
        self.features2 = np.empty((batchsize_max, FEATURES2_NUM, 9, 9), dtype=np.float32)
        # 推論の実行（初回は実行に時間が掛かる事があるため、ここで慣らし実行しておく）
        self.move_infer_choice(board)
    
    # 局面の推論
    def move_infer_choice(self, board):
        # 合法手生成
        color: int = board.turn
        legalmoves = np.array(list(board.legal_moves), dtype=np.uint32)
        # 合法手がない
        if len(legalmoves) == 0:
            return None
        # 入力特徴量の作成
        make_input_features(board, self.features1[0], self.features2[0])
        # 推論
        batchsize = 1
        io_binding = self.session.io_binding()
        io_binding.bind_cpu_input('input1', self.features1[0:batchsize])
        io_binding.bind_cpu_input('input2', self.features2[0:batchsize])
        io_binding.bind_output('output_policy')
        io_binding.bind_output('output_value')
        self.session.run_with_iobinding(io_binding)
        # 推論結果の取り出し
        infer_policy_logits, infer_values = io_binding.copy_outputs_to_cpu()
        infer_policy_logit = infer_policy_logits[0]
        # 合法手ごとの選択強度
        probabilities_logit = np.empty(len(legalmoves), dtype=np.float32)
        for i in range(len(legalmoves)):
            move = legalmoves[i]
            move_label = make_move_label(move, color)
            probabilities_logit[i] = infer_policy_logit[move_label]
        # Boltzmann分布
        probabilities = self.softmax_temperature_with_normalize(probabilities_logit)
        sfendict = dict()
        for i in range(len(legalmoves)):
            sfendict[move_to_usi(legalmoves[i])] = round(probabilities[i]*100.0, 1)
        return sfendict

    # 正規化
    def softmax_temperature_with_normalize(self, logits):
        # 確率を計算(オーバーフローを防止するため最大値で引く)
        max_logit = max(logits)
        probabilities = np.exp(logits - max_logit)
        # 合計が1になるように正規化
        sum_probabilities = sum(probabilities)
        probabilities /= sum_probabilities
        return probabilities

    
#コマンド受付
def command():
    while True:
        cmdline = input()
        # 局面設定
        if cmdline[:8] == "position":
            board.set_position(cmdline[9:])
        if cmdline[:2] == "go":
            policy_view(board)
        #終了処理
        if cmdline[:4] == "quit":
            root.destroy()
            sys.exit()
        usi(cmdline)

#コマンド入力処理
def usi(command):
    shogi.stdin.write(command+"\n")
    shogi.stdin.flush()

#コマンド出力処理
def output():
    while True:
        #エンジンからの出力を受け取る
        line = shogi.stdout.readline()
        #評価値バー処理
        shogibar(line)
        #標準出力
        sys.stdout.write(line)
        sys.stdout.flush()

#推定選択率更新
def policy_view(board):
    global sfenlist2
    sfenlist = ["推定選択率上位\n","\n","\n","\n","\n","\n","\n","\n"]
    # 選択率を計算
    sfendict = player.move_infer_choice(board)
    if sfendict == None:
        suitei["text"] = "".join(sfenlist)
        return
    sfenpack = sorted(sfendict.items(),key=lambda x:x[1],reverse=True)
    #選択率上位書き出し
    n = 0
    for x,y in sfenpack:
        sfenki2 = KI2.move_to_ki2(board.move_from_usi(x), board)
        sfenlist[n+1] = sfenki2+"("+str(y)+"%)"+"\n"
        n += 1
        if n >= 7:
            break
    suitei["text"] = "".join(sfenlist)

#評価値バー情報更新
def shogibar(line):
    if line[:4] == "info":
        if board.is_game_over():
            return
        sfen = line.split()
        #評価値表示列かチェック
        if "multipv" not in sfen and "pv" in sfen:
            pass
        elif "multipv" in sfen and int(sfen[sfen.index("multipv")+1]) == 1:
            pass
        else:
            return
        #左右反転チェックを受け取る
        if bln.get():
            reverse = -1
            ltebanlabel["text"] = player2
            rtebanlabel["text"] = player1
        else:
            reverse = 1
        #データ処理
        turn = -(board.turn * 2 - 1) * reverse
        move_count = board.move_number
        depth = int(sfen[sfen.index("depth")+1])
        if "nodes" in sfen:
            nodes = int(sfen[sfen.index("nodes")+1])
        else:
            nodes = 1
        if nodes < 10000:
            nodes = str(nodes) + "局面"
        elif nodes < 100000000:
            nodes = str(int(nodes/10000)) + "万局面"
        else:
            nodes = str(int(nodes/100000000)) + "億局面"                    
        pv = sfen[sfen.index("pv")+1]
        if "cp" in sfen:
            cp = int(sfen[sfen.index("cp")+1]) * turn
            #勝率変換(Ponanza定数 = 1200)
            winrate = int(round(100 / (1 + (np.exp(-cp/1200)))))
            lwinratelabel["text"] = str(winrate) + "%(" + "{:+}".format(cp) + ")"
            rwinratelabel["text"] = "(" + "{:+}".format(-cp) + ")" + str(100 - winrate) + "%"
            leftgraph.place(x=50, y=40, width=winrate * 15, height=20)
        if "mate" in sfen:
            mate = int(sfen[sfen.index("mate")+1]) * turn
            if mate > 0:
                lwinratelabel["text"] = "100%(" + str(mate) + "手詰)"
                rwinratelabel["text"] = "(" + str(-mate) + "手詰)0%"
                leftgraph.place(x = 50, y = 40, width = 1500, height = 20)
            elif mate < 0:
                lwinratelabel["text"] = "0%(" + str(mate) + "手詰)"
                rwinratelabel["text"] = "(" + str(-mate) + "手詰)100%"
                leftgraph.place(x = 50, y = 40, width = 0, height = 20)
        if turn == 1:
            lturnlabel["text"] = "▶"
            rturnlabel["text"] = ""
        else:
            rturnlabel["text"] = "◀"
            lturnlabel["text"] = ""
        saizen["text"] = engine + " " + str(move_count) + "手目検討中：最善手" + KI2.move_to_ki2(board.move_from_usi(pv), board)
        tansaku["text"] = "探索深度：" + str(depth) + "手 探索局面数：" + nodes
    

#コマンド受付と出力は並列処理(Tkinterとは別に動かす必要があるため)
t1 = threading.Thread(target=output, daemon=True)
t1.start()

#初期設定(isreadyまで)
while True:
    cmdline = input()
    if cmdline[:7] == "isready":
        # 推論処理初期化
        board.reset()
        player = OnnxPolicyPlayer(modelfile)
        usi(cmdline)
        break
    elif cmdline[:4] == "quit":        
        sys.exit()
    usi(cmdline)
    
#isready後は並列処理
t2 = threading.Thread(target=command, daemon=True)
t2.start()

#Tkinter表示
root = tk.Tk()
root.geometry("1100x330")
root.minsize(width=1600, height=330)
root.title("SuishoInfo")
root.configure(bg = bgcolor)


#勝率ラベル
lwinratelabel = tk.Label(root, text="50%(0)", font=(barfont, 25), bg=bgcolor, fg=fgcolor)
lwinratelabel.place(x = 50, y = 20, anchor=tk.W)
rwinratelabel = tk.Label(root, text="(0)50%", font=(barfont, 25), bg=bgcolor, fg=fgcolor)
rwinratelabel.place(x = 1550, y = 20, anchor=tk.E)

#手番ラベル
ltebanlabel = tk.Label(root, text=player1, font=(barfont, 25), bg=bgcolor, fg=turnfgcolor)
ltebanlabel.place(x = 50, y = 80, anchor=tk.W)
rtebanlabel = tk.Label(root, text=player2, font=(barfont, 25), bg=bgcolor, fg=turnfgcolor)
rtebanlabel.place(x=1550, y=80, anchor=tk.E)
lturnlabel = tk.Label(root, text="▶", font=(barfont, 25), bg=bgcolor, fg=turnfgcolor)
lturnlabel.place(x = 25, y = 80, anchor=tk.W)
rturnlabel = tk.Label(root, text="◀", font=(barfont, 25), bg=bgcolor, fg=turnfgcolor)
rturnlabel.place(x=1575, y=80, anchor=tk.E)

#最善手ラベル
saizen = tk.Label(root, text=engine + " 0手目検討中：最善手", font=(barfont, 20), bg=bgcolor, fg=fgcolor2)
saizen.place(x = 797, y = 18, anchor=tk.CENTER)

#探索ラベル
tansaku = tk.Label(root, text="探索深度：0手 探索局面数：0局面", font=(barfont, 20), bg=bgcolor, fg=fgcolor3)
tansaku.place(x=797, y=82, anchor=tk.CENTER)

#勝率目盛り
label50 = tk.Label(root, text="", bg="black")
label50.place(x=797, y=35, width=3, height=30)

#評価値バー描画
rightgraph = tk.Label(root, text="", bg=rightgraphbg, relief=tk.SOLID, bd=3)
rightgraph.place(x=50, y=40, width=1500, height=20)
leftgraph = tk.Label(root, text="", bg=leftgraphbg, relief=tk.SOLID, bd=3)
leftgraph.place(x = 50, y = 40, width = 750, height = 20)

#推定選択率
suiteibg = tk.Label(root, text="", bg=bgcolor)
suiteibg.place(x = 10, y = 115, width = 350, height = 230)
suitei = tk.Label(root, text="推定選択率上位", justify="left", font=(barfont, 18), bg=bgcolor, fg=fgcolor)
suitei.place(x=30, y=120, anchor=tk.NW)

#左右反転チェック
bln = tk.BooleanVar()
bln.set(False)
check = tk.Checkbutton(root, variable=bln, text="Reverse", font=(barfont, 15), bg=bgcolor, fg=fgcolor, selectcolor=bgcolor, activeforeground = fgcolor)
check.place(x = 500, y = 280)

root.mainloop()
