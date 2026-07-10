import os
import sys
import json
import heapq
import pandas as pd
from pathlib import Path
import plot_flow_map

file_dir = Path(__file__).resolve().parent
# 1階層上の親ディレクトリにある linehaul-problem を指す場合
linehaul_folder_path = os.path.join(file_dir.parent.parent, 'linehaul-problem')
if linehaul_folder_path not in sys.path:
    sys.path.append(linehaul_folder_path)

try:
    from main import main as evaluate_main
except ImportError:
    print("エラー: linehaul-problem/main.py から main 関数をインポートできませんでした。パスを確認してください。")
    sys.exit(1)

def solve_kyushu_backward_dijkstra(method_name, sum_multi):
    # パスの設定（スクリプトと同じディレクトリにあるcsvを読み込む想定）
    route_file_path = os.path.join(file_dir, 'truck_distance_time_long.csv')
    prefectures_file_path = os.path.join(file_dir, 'prefectures.csv')
    
    
    if not os.path.exists(route_file_path):
        print(f"エラー: {route_file_path} が見つかりません。")
        return

    # 2. データの読み込み
    df_route = pd.read_csv(route_file_path)
    df_pref = pd.read_csv(prefectures_file_path)
    PREFECTURES_CODE = df_pref["prefecture_code"]

    
    # 3. 逆向きグラフ（Backward Graph）の構築
    # 通常のリンク u -> v (コスト w) に対し、backward_graph[v] に (u, w) を入れる
    backward_graph = {code: [] for code in PREFECTURES_CODE}
    for _, row in df_route.iterrows():
        u = int(row['origin_prefecture_code'])
        v = int(row['destination_prefecture_code'])
        cost = float(row['time_min'])  # コンペのコスト定義に基づき、移動時間(分)を重みとする
        
        backward_graph[v].append((u, cost))
        
    # 解を格納する辞書 (発地, 着地) -> next_hop
    next_hop_map = {}

    # そのルートを通る荷量
    N = len(PREFECTURES_CODE)
    code_to_idx = {code: i for i, code in enumerate(PREFECTURES_CODE)}
    route_check = [[False] * N for _ in range(N)]
    
    
    # 4. 各着地（d）を起点とした後退ダイクストラ法の実行
    for d in PREFECTURES_CODE:
        # 最短距離の初期化
        dist = {code: float('inf') for code in PREFECTURES_CODE}
        # 逆向き探索における「直前のノード」（＝順向きにおけるNext-Hop）
        parent_node = {code: None for code in PREFECTURES_CODE}
        
        dist[d] = 0
        pq = [(0.0, d)]  # (累積コスト, 現在のノード)
        
        while pq:
            current_cost, v = heapq.heappop(pq)
            
            if current_cost > dist[v]:
                continue
                
            # 逆向きのエッジをたどる (v ➔ u つまり、順方向は u ➔ v)
            for u, weight in backward_graph[v]:
                # 通ったことのないルートなら、既存の重み
                if (not route_check[code_to_idx[u]][code_to_idx[v]] 
                    and dist[v] + weight < dist[u]):
                    dist[u] = dist[v] + weight
                    parent_node[u] = v  # uからdへ行くための次のホップはvになる
                    route_check[code_to_idx[u]][code_to_idx[v]] = True
                    heapq.heappush(pq, (dist[u], u))
                # 既存のルートなら、重みを低減
                elif (route_check[code_to_idx[u]][code_to_idx[v]] 
                      and dist[v] + sum_multi * weight < dist[u]):
                    dist[u] = dist[v] + sum_multi * weight
                    parent_node[u] = v  # uからdへ行くための次のホップはvになる
                    heapq.heappush(pq, (dist[u], u))
                    
        # 5. 求まったNext-Hopをマッピングに記録
        for o in PREFECTURES_CODE:
            if o == d:
                # 自身への移動は、仕様通り自身のコードにする
                next_hop_map[(o, d)] = o
            else:
                # ダイクストラ法で求まった次のノードを格納
                # 万が一、経路が繋がっていない場合は直送(d)をフォールバックとする
                next_hop_map[(o, d)] = parent_node[o] if parent_node[o] is not None else d

    # 6. コンペの提出フォーマット（発地昇順 ➔ 着地昇順 の1次元リスト）に整形
    output_solution = []
    for o in PREFECTURES_CODE:
        for d in PREFECTURES_CODE:
            output_solution.append(next_hop_map[(o, d)])
            
    # 7. ★【修正】7要素ごとに改行を入れてJSON風に整形して出力
    output_json_path = os.path.join(file_dir, method_name, 'input.json')
    
    # 7要素ずつの塊（行）を作り、カンマ区切りの文字列にする
    json_lines = []
    for i in range(0, len(output_solution), N):
        row_chunk = output_solution[i : i + N]
        # [40, 41, 41, ...] を "40, 41, 41, ..." という文字列にする
        row_str = ", ".join(map(str, row_chunk))
        json_lines.append(f"  {row_str}")
    output_string = "[\n" + ",\n".join(json_lines) + "\n]"
    
    # 全体を大カッコ [ ] で囲み、各行をカンマ＋改行で結合する
    with open(output_json_path, 'w', encoding='utf-8') as f:
        f.write(output_string)
        
    print(f"正常に最短経路の解を作成しました: {output_json_path}")
    print(f"要素数: {len(output_solution)} (7x7)")
    print(f"生成された解: {output_solution}")

if __name__ == '__main__':
    method_name = "dijkstra001"
    sum_route_multi = 0.01
    solve_kyushu_backward_dijkstra(method_name, sum_route_multi)
    plot_flow_map.main(method_name)
    evaluate_main("kyuusyuu", "test", method_name)
