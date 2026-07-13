from pathlib import Path
import pandas as pd
from pathlib import Path
import os

# 1. スクリプトファイル自体の絶対パスを取得し、その親ディレクトリ（parent）を取得
file_dir = Path(__file__).resolve().parent
prefectures_path = os.path.join(file_dir ,"prefectures.csv")
flows_path = os.path.join(file_dir, "virtual_prefecture_flows.csv")
distance_path = os.path.join(file_dir, "truck_distance_time_long.csv")

# 1. 3つのCSVデータを読み込み
df_pref = pd.read_csv(prefectures_path)
df_flows = pd.read_csv(flows_path)
df_dist = pd.read_csv(distance_path)

# prefectures.csv に含まれる有効な都道府県コードのセットを取得（九州7県: 40〜46）
valid_codes = set(df_pref["prefecture_code"])

# 3. truck_distance_time_long.csv から移動時間(time_min)を引くための辞書を作成
# 鍵: (発地コード, 着地コード) -> 値: 移動時間(分)
time_map = df_dist.set_index(["origin_prefecture_code", "destination_prefecture_code"])["time_min"].to_dict()

# 4. 全便を直通とした場合の合計コストを計算
total_cost = 0.0
calculated_count = 0

for _, row in df_flows.iterrows():
    orig_code = int(row["origin_prefecture_code"])
    dest_code = int(row["destination_prefecture_code"])
    weight_kg = row["virtual_load"]  # 輸送量 (kg)
    truck_num = weight_kg // 100000 + 1
    # 同一県内の移動（例: 福岡発・福岡着）の場合
    if orig_code == dest_code:
        minute = 0.0  # 移動時間は0分として計算
    else:
        # 辞書から実移動時間(time_min)を取得。データがない場合は安全のため0.0
        minute = time_map.get((orig_code, dest_code), 0.0)
        
    # コスト計算式: (kg) * (minute) / 60 / 10
    cost = (weight_kg * minute) / 60 /10
    cost = truck_num * minute / 60 * 10000
    total_cost += cost
    calculated_count += 1

# 5. 結果の出力
print(f"--- 九州ケース 直通コスト計算結果 ---")
print(f"------------------------------------")
print(f"対象となった総流動数 : {calculated_count} 件 ")
print(f"calc_limit_minimal   : {total_cost:,.2f}")
