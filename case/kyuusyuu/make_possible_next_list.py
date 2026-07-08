import os
import json
import pandas as pd

current_dir = os.getcwd()
route_file_path = 'case/test/truck_distance_time_long.csv'
df_route = pd.read_csv(route_file_path)

# 都道府県ごとにグループ化し、それぞれの到達先リストを辞書（Dict）にする
next_map = {}
for pref_name, group in df_route.groupby('origin_prefecture_name'):
    next_map[pref_name] = group['destination_prefecture_name'].unique().tolist()

# JSONファイルとして現在のフォルダに保存
save_path_json = os.path.join(current_dir, 'all_prefectures_next_map.json')
with open(save_path_json, 'w', encoding='utf-8') as f:
    json.dump(next_map, f, ensure_ascii=False, indent=4)

print(f"全都道府県の到達先マップをJSONとして保存しました:\n{save_path_json}")