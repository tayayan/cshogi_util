import numpy as np
from cshogi import *

psvfile = input("PSVファイルのパスを入力してね")
output = input("outputファイルの名前は？")

psva = np.fromfile(psvfile, dtype=PackedSfenValue)
psvb = np.zeros(len(psva), dtype=PackedSfenValue)

i = 0
for psv in psva:
    if abs(psv['score']) > 200:
        continue
    psvb[i]["sfen"] = psv["sfen"]
    psvb[i]["move"] = psv["move"]
    psvb[i]["score"] = psv["score"]
    psvb[i]["gamePly"] = psv["gamePly"]
    psvb[i]["game_result"] = psv["game_result"]        
    i += 1

psvb[:i].tofile(output)

print("完了！")
input()
