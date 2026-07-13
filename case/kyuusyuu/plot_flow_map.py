import os
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main(method_name = None, save_name = None):
    # 1. 各ファイルの読み込み形式（plot_load_map.pyの形式に完全準拠）
    file_dir = Path(__file__).resolve().parent
    if method_name !=  None:
        case_dir = os.path.join(file_dir, method_name)
    else:
        case_dir = file_dir
    pref_file_path = os.path.join(file_dir, 'prefectures.csv')
    route_file_path = os.path.join(file_dir, 'truck_distance_time_long.csv')
    flow_file_path = os.path.join(file_dir, 'virtual_prefecture_flows.csv')
    input_file_path = os.path.join(case_dir, 'input.json')

    # 必要ファイルの存在チェック
    for p in [pref_file_path, route_file_path, flow_file_path, input_file_path]:
        if not os.path.exists(p):
            print(f"エラー: {os.path.basename(p)} が見つかりません。")
            return

    df_pref = pd.read_csv(pref_file_path)
    df_route = pd.read_csv(route_file_path)
    df_flow = pd.read_csv(flow_file_path)
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # 都道府県コードのリストを取得し、評価器と同じ順序（昇順）にソート
    nodes = sorted(df_pref['prefecture_code'].unique())
    n_nodes = len(nodes)

    if len(input_data) != n_nodes * n_nodes:
        print(f"エラー: input.json の要素数({len(input_data)})が、都道府県数の二乗({n_nodes * n_nodes})と一致しません。")
        return

    # 2. Next-Hopテーブルの構築 (発ノード × 着ノード)
    next_hop_table = {}
    idx = 0
    for f_node in nodes:
        for t_node in nodes:
            next_hop_table[(f_node, t_node)] = input_data[idx]
            idx += 1

    # 3. 経路の復元と各リンクの流量（virtual_load）の集計
    link_flows = {}  # key: (from_code, to_code), value: total_load
    
    for _, row in df_flow.iterrows():
        orig = int(row['origin_prefecture_code'])
        dest = int(row['destination_prefecture_code'])
        load = float(row['virtual_load'])
        
        if orig == dest:  # 自拠点内の流動は幹線を通らないためスキップ
            continue
            
        curr = orig
        visited = {curr}
        loop_error = False
        
        # 目的地に到達するまでNext-Hopを辿る
        while curr != dest:
            nxt = next_hop_table.get((curr, dest))
            if nxt is None:
                print(f"警告: ({curr}, {dest}) のNext-Hopが登録されていません。")
                loop_error = True
                break
            if nxt in visited:
                print(f"警告: 発{orig}➔着{dest}の経路中、ノード{nxt}で循環（無限ループ）を検知したため集計をスキップします。")
                loop_error = True
                break
                
            # 有向リンク(curr ➔ nxt) に流量を加算
            link_flows[(curr, nxt)] = link_flows.get((curr, nxt), 0.0) + load
            visited.add(nxt)
            curr = nxt
            
        if loop_error:
            continue

    # 4. グラフ描画の設定
    coord_dict = df_pref.set_index('prefecture_code')[['longitude', 'latitude']].to_dict('index')
    fig, ax = plt.subplots(figsize=(12, 10))

    # --- ① 経路の描画 (線: グレーの薄い背景線 - 完全準拠) ---
    possible_way_count = 0
    for idx, row in df_route.iterrows():
        orig_code = row['origin_prefecture_code']
        dest_code = row['destination_prefecture_code']
        
        if orig_code in coord_dict and dest_code in coord_dict:
            x = [coord_dict[orig_code]['longitude'], coord_dict[dest_code]['longitude']]
            y = [coord_dict[orig_code]['latitude'], coord_dict[dest_code]['latitude']]
            ax.plot(x, y, color='gray', alpha=0.3, linewidth=0.5, zorder=1)
            possible_way_count += 1

    # --- ② 流量のある路線の強調描画（矢印化・並行分離）と数値の記入 ---
    max_flow = max(link_flows.values()) if link_flows else 1.0

    for (f_code, t_code), flow in link_flows.items():
        if flow <= 0:
            continue
            
        if f_code in coord_dict and t_code in coord_dict:
            x1, y1 = coord_dict[f_code]['longitude'], coord_dict[f_code]['latitude']
            x2, y2 = coord_dict[t_code]['longitude'], coord_dict[t_code]['latitude']
            
            # 流量に応じて線の太さを動的に変更 (最小0.8px 〜 最大5.0px)
            lw = 0.8 + (flow / max_flow) * 4.2
            
            # 方向ベクトルと距離の計算
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist > 0:
                # 進行方向の「右側」を指す単位法線ベクトル
                nx = dy / dist
                ny = -dx / dist
                
                # 【改良1】双方向の重複を防ぐため、線自体を右側に少しオフセット（スライド）させる
                line_offset = 0.015  
                x1_shifted = x1 + nx * line_offset
                y1_shifted = y1 + ny * line_offset
                x2_shifted = x2 + nx * line_offset
                y2_shifted = y2 + ny * line_offset
                
                # 【改良2】大きな拠点コード(fontsize=20)やプロットと被らないよう、矢印の始点と終点を少し内側に縮める
                shrink_dist = 0.07  # ノード手前で止める距離
                ux = dx / dist
                uy = dy / dist
                
                if dist > shrink_dist * 2:
                    x1_arr = x1_shifted + ux * shrink_dist
                    y1_arr = y1_shifted + uy * shrink_dist
                    x2_arr = x2_shifted - ux * shrink_dist
                    y2_arr = y2_shifted - uy * shrink_dist
                else:
                    x1_arr, y1_arr = x1_shifted, y1_shifted
                    x2_arr, y2_arr = x2_shifted, y2_shifted
                
                # 実際に荷物が流れるルートを有向矢印（dodgerblue）で描画
                # 線の太さに合わせて矢印の頭のサイズ（mutation_scale）も動的に調整
                ax.annotate("", xy=(x2_arr, y2_arr), xytext=(x1_arr, y1_arr),
                            arrowprops=dict(arrowstyle="-|>", color='dodgerblue', alpha=0.8, 
                                            linewidth=lw, mutation_scale=12 + lw),
                            zorder=2)
                
                # テキストを配置する座標（分離した矢印線のさらに右側に配置して視認性を確保）
                text_offset = 0.045
                tx = ((x1 + x2) / 2) + nx * text_offset
                ty = ((y1 + y2) / 2) + ny * text_offset
                
                # 流量のテキスト
                flow_text = f"{int(flow):,}"
                
                # 白背景のボックス付きで数値を印字
                ax.text(tx, ty, flow_text, color='darkred', fontsize=10, 
                        fontweight='bold', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='gray', alpha=0.85, linewidth=0.3),
                        zorder=4)

    # --- ③ 都道府県のプロット (点 - 完全準拠) ---
    ax.scatter(df_pref['longitude'], df_pref['latitude'], color='blue', edgecolors='black', s=50, zorder=3)

    # --- ④ 都道府県コードの描画 (ラベル - 完全準拠形式) ---
    for idx, row in df_pref.iterrows():
        ax.text(row['longitude'] + 0.02, row['latitude'] + 0.02, row['prefecture_code'], fontsize=20, zorder=5)

    # 5. グラフの装飾設定
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6, zorder=2)
    ax.set_aspect('equal', adjustable='box')

    # 6. グラフの保存と表示
    if save_name !=  None:
        case_dir = os.path.join(file_dir, save_name)
    save_path = os.path.join(case_dir, 'route_flow_map.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"正常に有向流量マップ画像を保存しました: {save_path}")

    # plt.show()

if __name__ == '__main__':
    method_name = "direct"
    main(method_name)
