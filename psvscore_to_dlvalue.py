#やねうら王教師局面の評価値をdlshogiモデルのvalueに書き換えるプログラム
import numpy as np
from cshogi import *
from cshogi.dlshogi import make_input_features, make_move_label, FEATURES1_NUM, FEATURES2_NUM
import onnxruntime

model_path = "model.onnx"
device_id = 0
enable_cuda = True
enable_tensorrt = True

input_features = [
    np.empty((512, FEATURES1_NUM, 9, 9), dtype=np.float32),
    np.empty((512, FEATURES2_NUM, 9, 9), dtype=np.float32),
]

#局面
board = Board()

#推論セッション
available_providers = onnxruntime.get_available_providers()
enable_providers = []
if enable_tensorrt and 'TensorrtExecutionProvider' in available_providers:
    enable_providers.append(('TensorrtExecutionProvider', {
        'device_id': device_id,
        'trt_fp16_enable': True,
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


#勝率を評価値にする
def value_to_eval(value):
    if value == 1.0:
        return 32000
    if value == 0.0:
        return -32000
    else:
        return int(-600 * np.log(1/value - 1))


#PSV読み込み 1000万局面×800回
a = 0
while a != 800:
    psva = np.fromfile("psv.bin", count=10000000, offset=10000000*a*40, dtype=PackedSfenValue)
    psvb = np.zeros(len(psva), dtype=PackedSfenValue)
    scores = np.array([], dtype=np.float32)

    j = 0
    for i in range(10000000):
        psvb[i]["sfen"] = psva[i]["sfen"]
        psvb[i]["move"] = psva[i]["move"]
        psvb[i]["gamePly"] = psva[i]["gamePly"]
        psvb[i]["game_result"] = psva[i]["game_result"]
        
        board.set_psfen(psva[i]['sfen'])
        make_input_features(board, input_features[0][j], input_features[1][j])
        #500局面貯まったら推論
        if j == 499:
            io_binding = session.io_binding()
            io_binding.bind_cpu_input("input1", input_features[0][:j + 1])
            io_binding.bind_cpu_input("input2", input_features[1][:j + 1])
            io_binding.bind_output("output_policy")
            io_binding.bind_output("output_value")
            session.run_with_iobinding(io_binding)
            outputs = io_binding.copy_outputs_to_cpu()
            values = outputs[1].reshape(-1)
            scores = np.append(scores, values)
            j = 0
        else:
            j += 1
        if (i + 1) % 1000000 == 0:
            print(str(i)+"局面処理")

    #評価値をつける
    print("評価値追加中…")
    for psv, score in zip(psvb, scores):
        psv["score"] = value_to_eval(score)

    psvb.tofile("psv" + str(a) + ".bin")
    print("psv" + str(a) + ".bin作成完了")
    a += 1
