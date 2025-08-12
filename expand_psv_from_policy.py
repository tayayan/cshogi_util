import numpy as np
from cshogi import *
from cshogi.dlshogi import make_input_features, make_move_label, FEATURES1_NUM, FEATURES2_NUM
import onnxruntime
import datetime
import multiprocessing

model_path = "model.onnx"
device_id = 0
enable_cuda = True
enable_tensorrt = True

def softmax_temperature_with_normalize(logits):
    """
    ロジットを正規化された確率に変換します。
    この関数は各ワーカープロセスから呼ばれるため、グローバルスコープに配置します。
    """
    # 確率を計算(オーバーフローを防止するため最大値で引く)
    max_logit = np.max(logits)
    probabilities = np.exp(logits - max_logit)
    # 合計が1になるように正規化
    sum_probabilities = np.sum(probabilities)
    probabilities /= sum_probabilities
    return probabilities

def process_chunk(psv, policy):
    """
    単一の局面(psv)とポリシー(policy)を処理し、
    選択率が10%を超える手によって生成される次局面のPSFEN文字列のリストを返します。
    """
    board = Board()
    board.set_psfen(psv['sfen'])
    color: int = board.turn
    legal_moves = np.array(list(board.legal_moves), dtype=np.uint32)

    if len(legal_moves) == 0:
        return []

    # 合法手に対応するポリシーの値を抽出
    probabilities_logit = np.empty(len(legal_moves), dtype=np.float32)
    for i, move in enumerate(legal_moves):
        move_label = make_move_label(move, color)
        probabilities_logit[i] = policy[move_label]

    # 確率に変換
    probabilities = softmax_temperature_with_normalize(probabilities_logit)

    # 選択率が10%を超える手を探し、次局面のSFENを生成
    next_sfens = []
    for i, move in enumerate(legal_moves):
        if probabilities[i] * 100.0 > 10.0:
            board.push(move)
            next_sfens.append(board.sfen())
            board.pop()
    
    return next_sfens


def main():
    """
    メインの処理を実行します。
    """
    f = open("newpsv.bin", "ab")

    input_features = [
        np.empty((1024, FEATURES1_NUM, 9, 9), dtype=np.float32),
        np.empty((1024, FEATURES2_NUM, 9, 9), dtype=np.float32),
    ]

    # --- 局面と推論セッションの初期化 ---
    board = Board()
    available_providers = onnxruntime.get_available_providers()
    enable_providers = []
    if enable_tensorrt and 'TensorrtExecutionProvider' in available_providers:
        enable_providers.append(('TensorrtExecutionProvider', {
            'device_id': device_id,
            'trt_fp16_enable': False,
            'trt_engine_cache_enable': True,
        }))
        print("Enable TensorrtExecutionProvider")
    if enable_cuda and 'CUDAExecutionProvider' in available_providers:
        enable_providers.append(('CUDAExecutionProvider', {
            'device_id': device_id,
        }))
        print("Enable CUDAExecutionProvider")
    if 'CPUExecutionProvider' in available_providers:
        enable_providers.append('CPUExecutionProvider')
        print("Enable CPUExecutionProvider")
    session = onnxruntime.InferenceSession(model_path, providers=enable_providers)

    # 16個のプロセスを持つプールを作成
    with multiprocessing.Pool(16) as pool:
        a = 0
        #76億2000万局面を処理
        while a != 762000:
            psvs = np.fromfile("oldpsv.bin", count=10000, offset=10000*a*40, dtype=PackedSfenValue)
            if psvs.size == 0:
                print("読み込みファイルの終端に達しました。処理を終了します。")
                break
            
            s = 0
            j = 0
            all_policies = [] # このループで推論されたポリシーをすべて保持するリスト
            
            # --- ミニバッチでの推論処理 ---
            num_psvs = len(psvs)
            for i in range(num_psvs):        
                board.set_psfen(psvs[i]['sfen'])
                make_input_features(board, input_features[0][j], input_features[1][j])
                j += 1
                
                # 1000局面貯まるか、最後の局面になったら推論
                if j == 1000 or i == num_psvs - 1:
                    io_binding = session.io_binding()
                    io_binding.bind_cpu_input("input1", input_features[0][:j])
                    io_binding.bind_cpu_input("input2", input_features[1][:j])
                    io_binding.bind_output("output_policy")
                    io_binding.bind_output("output_value")
                    session.run_with_iobinding(io_binding)
                    outputs = io_binding.copy_outputs_to_cpu()
                    
                    all_policies.append(outputs[0])
                    j = 0
            
            policies = np.concatenate(all_policies)

            # 選択率10%超えの局面を教師データ化する (16並列で実行)
            
            # 各プロセスに渡す引数のペアを作成
            process_args = zip(psvs, policies)
            
            # 並列処理を実行し、結果を収集
            # pool.starmapは、(psv, policy)のタプルを process_chunk(psv, policy) のように展開して呼び出します。
            results = pool.starmap(process_chunk, process_args)
            
            # 結果を平坦化: [[sfen1, sfen2], [], [sfen3]] -> [sfen1, sfen2, sfen3]
            flat_next_sfens = [sfen for sublist in results for sfen in sublist]

            k = len(flat_next_sfens)
            if k > 0:
                # PSFEN文字列のリストからPackedSfenValueのnumpy配列に変換
                newpsv = np.empty(k, dtype=PackedSfenValue)
                temp_board = Board() # 変換用のボード
                for i, sfen_str in enumerate(flat_next_sfens):
                    temp_board.set_sfen(sfen_str)
                    temp_board.to_psfen(newpsv[i]['sfen'])
                
                # ファイル書き込み
                newpsv.tofile(f)

            if a % 100 == 0:
                print(datetime.datetime.now())
                print(f"{a*10000}局面作成完了")
            a += 1

    f.close()
    print("すべての処理が完了しました。")

if __name__ == '__main__':
    main()
