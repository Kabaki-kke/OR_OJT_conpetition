import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 各ファイルの読み込み
# 実行する環境に合わせてパスを調整してください
file_dir = Path(__file__).resolve().parent
pref_file_path = os.path.join(file_dir, 'prefectures.csv')
route_file_path = os.path.join(file_dir, 'truck_distance_time_long.csv')

df_pref = pd.read_csv(pref_file_path)
df_route = pd.read_csv(route_file_path)

# 2. 都道府県コードから経度・緯度を高速に参照するための辞書を作成
coord_dict = df_pref.set_index('prefecture_code')[['longitude', 'latitude']].to_dict('index')

# 3. グラフ描画の設定
fig, ax = plt.subplots(figsize=(12, 10))

# --- ① 経路の描画 (線) ---
# 点や文字の後ろに隠れるよう zorder=1 に設定し、
# 線が密集しても見やすくなるよう透明度(alpha)と線の太さ(linewidth)を抑えています
possible_way_count = 0
for idx, row in df_route.iterrows():
    orig_code = row['origin_prefecture_code']
    dest_code = row['destination_prefecture_code']
    
    # 双方の座標データが存在する場合に線を引く
    if orig_code in coord_dict and dest_code in coord_dict:
        x = [coord_dict[orig_code]['longitude'], coord_dict[dest_code]['longitude']]
        y = [coord_dict[orig_code]['latitude'], coord_dict[dest_code]['latitude']]
        ax.plot(x, y, color='gray', alpha=0.3, linewidth=0.5, zorder=1)
        possible_way_count += 1

# --- ② 都道府県のプロット (点) ---
ax.scatter(df_pref['longitude'], df_pref['latitude'], color='blue', edgecolors='black', s=50, zorder=3)

# --- ③ 都道府県名の描画 (ラベル) ---
for idx, row in df_pref.iterrows():
    ax.text(row['longitude'] + 0.05, row['latitude'] + 0.05, row['prefecture_code'], fontsize=8, zorder=4)

# 4. グラフの装飾設定
ax.set_title(f'Prefecture Locations and Truck Routes (Count: {possible_way_count})', fontsize=14)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6, zorder=2)
ax.set_aspect('equal', adjustable='box')

# 5. グラフの保存
save_path = os.path.join(file_dir, 'prefectures_routes_map.png')
plt.savefig(save_path, bbox_inches='tight')

# ローカル環境の画面にポップアップ表示させたい場合は、以下のコメントアウトを解除してください
plt.show()
