from pathlib import Path
import pandas as pd
import os
from pathlib import Path

# 1. スクリプトファイル自体の絶対パスを取得し、その親ディレクトリ（parent）を取得
file_dir = Path(__file__).resolve().parent
prefectures_path = os.path.join(file_dir, "prefectures.csv")
flows_path = os.path.join(file_dir, "virtual_prefecture_flows.csv")
dist_path = os.path.join(file_dir, "truck_distance_time_long.csv")

# 1. 各CSVファイルを読み込み
df_pref = pd.read_csv(prefectures_path)
df_flows = pd.read_csv(flows_path)
df_dist = pd.read_csv(dist_path)

# 2. prefectures.csv に含まれる有効な都道府県コードをセットとして取得
valid_codes = set(df_pref["prefecture_code"])

# 3. 発地(origin)と着地(destination)の両方が valid_codes に含まれる行のみを抽出
filtered_flows = df_flows[
    df_flows["origin_prefecture_code"].isin(valid_codes)
    & df_flows["destination_prefecture_code"].isin(valid_codes)
]
filtered_dist = df_dist[
    df_dist["origin_prefecture_code"].isin(valid_codes)
    & df_dist["destination_prefecture_code"].isin(valid_codes)
]

# 4. 元のファイルに上書き保存 (BOM付きUTF-8)
filtered_flows.to_csv(flows_path, index=False, encoding="utf_8_sig")
filtered_dist.to_csv(dist_path, index=False, encoding="utf_8_sig")

print(
    f"書き換え完了：元の行数 {len(df_flows)} -> フィルタリング後 {len(filtered_flows)}"
)
