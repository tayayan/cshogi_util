from cshogi import *
from cshogi.dlshogi import FEATURES1_NUM, FEATURES2_NUM, make_input_features, make_move_label
import numpy as np
import onnxruntime
import subprocess

# License: GPLv3 https://www.gnu.org/licenses/gpl-3.0.html

# モデルファイル
modelfile = "model.onnx"
# エンジン
shogiengine = "YO.exe"

# 局面情報
board = Board()

# 推論クラス
player = None

#外部エンジン起動        
shogi = subprocess.Popen(shogiengine, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      encoding="cp932")

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


#コマンド入力処理
def usi(command):
    shogi.stdin.write(command+"\n")
    shogi.stdin.flush()

def sfen_to_winrate(sfen):
    
    def winrate(cp):
        winrate = int(round(100 / (1 + (np.exp(-cp/1200)))))
        return winrate

    usi("position sfen " + sfen)
    usi("go nodes 2000000")
    line = ""
    cp1 = -100000
    cp2 = 100000
    bestmove = ""
    while line[:8] != "bestmove":
        line = shogi.stdout.readline()
        if "multipv 1 " in line and "score cp" in line:
            info = line.split()
            cp1 = int(info[info.index("cp")+1])
        elif "multipv 2 " in line and "score cp" in line:
            info = line.split()
            cp2 = int(info[info.index("cp")+1])
    info = line.split()
    bestmove = info[info.index("bestmove")+1]
    rate1 = winrate(cp1)
    rate2 = winrate(cp2)
    return rate1, rate2, bestmove

player = OnnxPolicyPlayer()

usi("MultiPV 2")
usi("isready")

rfile = open(input("sfenファイルのパスを入力"),"r")
sfenline = "aaaa"

while sfenline != "":
    sfenline = rfile.readline()
    movelist = sfenline.split()[2:]
    for move in movelist:
        board.push_usi(move)
        if board.is_game_over() or board.is_nyugyoku():
            break
        rate1, rate2, bestmove = sfen_to_winrate(board.sfen())
        policydict = player.move_infer_choice(board)
        policy = policydict[bestmove]
        if 10 < rate1 - rate2 or (5 < rate1 - rate2 and rate1 > 50 and rate2 < 50):
            board.pop()
            print("sfen " + board.sfen() + " moves " + move)
            print("推定選択率：" + str(policy) + "%\n")
            board.push_usi(move)          
    board.reset()

rfile.close()
print("完了！")
