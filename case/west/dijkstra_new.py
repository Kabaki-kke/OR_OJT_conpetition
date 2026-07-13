import os
import sys
import json
import heapq
import math
import random
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

def calc_required_trucks_for_early_iteration(load, variance):
    if load <= 0:
        return 0
    sd = math.sqrt(variance)
    required_capacity = load #+ 1.65 * sd  # 荷量のブレを考慮した必要キャパ
    return max(1, required_capacity / 100000.0)  # 1台100,000kg制約を無視


# トラック台数を計算する関数（1台100,000kg、正規分布のブレを考慮）
def calc_required_trucks(load, variance):
    if load <= 0:
        return 0
    sd = math.sqrt(variance)
    required_capacity = load #+ 1.65 * sd  # 荷量のブレを考慮した必要キャパ
    return max(1, math.ceil(required_capacity / 100000.0))  # 1台100,000kg制約

def solve_kyushu_backward_dijkstra(method_name,input_case, max_iters = 100):
    # パスの設定
    route_file_path = os.path.join(file_dir, 'truck_distance_time_long.csv')
    prefectures_file_path = os.path.join(file_dir, 'prefectures.csv')
    flow_file_path = os.path.join(file_dir, 'virtual_prefecture_flows.csv')
    
    if not os.path.exists(route_file_path):
        print(f"エラー: {route_file_path} が見つかりません。")
        return

    # データの読み込み
    df_route = pd.read_csv(route_file_path)
    df_pref = pd.read_csv(prefectures_file_path)
    df_flow = pd.read_csv(flow_file_path)
    PREFECTURES_CODE = sorted(list(df_pref["prefecture_code"]))

    # 荷物量データを高速に引くための辞書化
    flow_map = df_flow.set_index(['origin_prefecture_code', 'destination_prefecture_code'])['virtual_load'].to_dict()
    sd_map = df_flow.set_index(['origin_prefecture_code', 'destination_prefecture_code'])['standard_deviation'].to_dict()

    # 各区間のベースとなる移動時間を保持する辞書
    base_time = {}
    backward_graph = {code: [] for code in PREFECTURES_CODE}
    for _, row in df_route.iterrows():
        u = int(row['origin_prefecture_code'])
        v = int(row['destination_prefecture_code'])
        time = float(row['time_min'])
        base_time[(u, v)] = time
        backward_graph[v].append((u, time))
        
    N = len(PREFECTURES_CODE)
    code_to_idx = {code: i for i, code in enumerate(PREFECTURES_CODE)}

    # ====================================================================
    # ステップ 1: 既存の input.json を読み込んで初期解 (init_next_hop) とする
    # ====================================================================
    init_next_hop = {}
    initial_json_path = os.path.join(file_dir,input_case, 'input.json') 
    
    if os.path.exists(initial_json_path):
        with open(initial_json_path, 'r', encoding='utf-8') as f:
            initial_data = json.load(f)
            
        idx = 0
        for o in PREFECTURES_CODE:
            for d in PREFECTURES_CODE:
                init_next_hop[(o, d)] = int(initial_data[idx])
                idx += 1
        print(f"✅ 既存の input.json を初期解として正常に読み込みました。")
    else:
        print(f"⚠️ {initial_json_path} が見つからないため、最短パスで初期解を自動生成します。")
        for d in PREFECTURES_CODE:
            dist = {code: float('inf') for code in PREFECTURES_CODE}
            parent_node = {code: None for code in PREFECTURES_CODE}
            dist[d] = 0
            pq = [(0.0, d)]
            while pq:
                current_cost, v = heapq.heappop(pq)
                if current_cost > dist[v]: continue
                for u, time in backward_graph[v]:
                    if dist[v] + time < dist[u]:
                        dist[u] = dist[v] + time
                        parent_node[u] = v
                        heapq.heappush(pq, (dist[u], u))
            for o in PREFECTURES_CODE:
                if o == d: init_next_hop[(o, d)] = o
                else: init_next_hop[(o, d)] = parent_node[o] if parent_node[o] is not None else d

    # ====================================================================
    # ステップ 2: 初期解に基づいて、現在の各区間のベース荷物量を集計
    # ====================================================================
    route_amount = [[0.0] * N for _ in range(N)]
    route_variance = [[0.0] * N for _ in range(N)]
    
    for _, row in df_flow.iterrows():
        orig, dest = int(row['origin_prefecture_code']), int(row['destination_prefecture_code'])
        load = float(row['virtual_load'])
        sd = float(row['standard_deviation'])
        if orig == dest or orig not in code_to_idx or dest not in code_to_idx: continue
        
        curr = orig
        visited = {curr}
        while curr != dest:
            nxt = init_next_hop[(curr, dest)]
            u_idx, v_idx = code_to_idx[curr], code_to_idx[nxt]
            route_amount[u_idx][v_idx] += load
            route_variance[u_idx][v_idx] += (sd ** 2)
            curr = nxt
            if curr in visited: break
            visited.add(curr)

    print(f"🚀 反復最適化を開始します（最大 {max_iters} 周）")
    
    for iter_idx in range(max_iters):
        # この周での Next-Hop の変更箇所数を記録するカウンター
        changes_in_this_iter = 0
        next_hop_map = init_next_hop.copy()
        easy_cost_check  =False

        if iter_idx <= max_iters:
            calculator = calc_required_trucks_for_early_iteration
            # changes_in_this_iter += 100
            # easy_cost_check = True  
        else:
            calculator = calc_required_trucks

        print(f"\n🔄 --- アウター・イテレーション {iter_idx + 1} / {max_iters} ---")
        
        # 1. 目的地リストをコピーしてシャッフル（振動対策）
        d_list = PREFECTURES_CODE.copy()
        random.shuffle(d_list)
        
        
        
        for d in d_list:
            
            # --- 🔄 随時更新【A】: 現在の着地 d 向けの古い流量を一旦マイナス ---
            for o in PREFECTURES_CODE:
                if o == d or o not in code_to_idx or d not in code_to_idx: continue
                load = flow_map.get((o, d), 0.0)
                sd = sd_map.get((o, d), 0.0)
                if load <= 0: continue
                
                curr = o
                visited = {curr}
                while curr != d:
                    nxt = init_next_hop[(curr, d)]
                    u_idx, v_idx = code_to_idx[curr], code_to_idx[nxt]
                    route_amount[u_idx][v_idx] = max(0.0, route_amount[u_idx][v_idx] - load)
                    route_variance[u_idx][v_idx] = max(0.0, route_variance[u_idx][v_idx] - (sd ** 2))
                    curr = nxt
                    if curr in visited: break
                    visited.add(curr)

            # --- 🔍 本番ダイクストラ法（1ノード確定ごとにroute_amountを更新） ---
            cost = {code_from: {code_curr: float('inf') for code_curr in PREFECTURES_CODE} for code_from in PREFECTURES_CODE}
            parent_node = {code: None for code in PREFECTURES_CODE}
            cost[d][d] = 0  # ➔ 💡 前回答の通り、[:] ではなく [d] に修正
            pq = [(0.0, d)]
            
            while pq:
                current_cost, v = heapq.heappop(pq)
                if current_cost > cost[v][v]: continue
                
                if v != d:
                    L_v = flow_map.get((v, d), 0.0)
                    V_v = (sd_map.get((v, d), 0.0)) ** 2
                    if L_v > 0:
                        curr_path = v
                        visited_in_path = set()
                        while curr_path != d:
                            if curr_path in visited_in_path: break
                            visited_in_path.add(curr_path)
                            
                            nxt_path = parent_node[curr_path]
                            if nxt_path is None: break
                            p_u_idx, p_v_idx = code_to_idx[curr_path], code_to_idx[nxt_path]
                            route_amount[p_u_idx][p_v_idx] += L_v
                            route_variance[p_u_idx][p_v_idx] += V_v
                            curr_path = nxt_path
                
                path_edges = []
                curr_path = v
                visited_in_path = set()
                while curr_path != d:
                    if curr_path in visited_in_path: break
                    visited_in_path.add(curr_path)
                    
                    nxt_path = parent_node[curr_path]
                    if nxt_path is None: break
                    path_edges.append((curr_path, nxt_path))
                    curr_path = nxt_path
                    
                for u, time in backward_graph[v]:
                    u_idx, v_idx = code_to_idx[u], code_to_idx[v]
                    
                    # ➔ 💡 ループチェック
                    is_loop = False
                    check_curr = v
                    check_visited = set()
                    while check_curr != d and check_curr is not None:
                        if check_curr == u or check_curr in check_visited:
                            is_loop = True
                            break
                        check_visited.add(check_curr)
                        check_curr = parent_node[check_curr]
                    
                    if is_loop: continue
                    
                    L = flow_map.get((u, d), 0.0)
                    V = (sd_map.get((u, d), 0.0)) ** 2
                    
                    tb_uv = calculator(route_amount[u_idx][v_idx], route_variance[u_idx][v_idx])
                    ta_uv = calculator(route_amount[u_idx][v_idx] + L, route_variance[u_idx][v_idx] + V)
                    total_incremental_cost = (ta_uv - tb_uv) * (10000.0 * time / 60.0) + time
                    total_curr_cost = 0
                    for edge_u, edge_v in path_edges:
                        eu_idx, ev_idx = code_to_idx[edge_u], code_to_idx[edge_v]
                        edge_time = base_time[(edge_u, edge_v)]
                        tb_edge = calculator(route_amount[eu_idx][ev_idx], route_variance[eu_idx][ev_idx])
                        ta_edge = calculator(route_amount[eu_idx][ev_idx] + L, route_variance[eu_idx][ev_idx] + V)
                        total_curr_cost = (ta_edge - tb_edge) * (10000.0 * edge_time / 60.0)
                    
                    cost[u][v] = total_curr_cost
                    if cost[u][v] + total_incremental_cost < cost[u][u]:
                        cost[u][u] = cost[u][v] + total_incremental_cost
                        parent_node[u] = v
                        heapq.heappush(pq, (cost[u][u], u))
                        
            # 2. この目的地 d に対する決定を記録し、前回の決定から変化したかチェック
            for o in PREFECTURES_CODE:
                old_hop = init_next_hop[(o, d)]
                new_hop = parent_node[o] if parent_node[o] is not None else d
                if o == d: new_hop = o
                
                if old_hop != new_hop:
                    changes_in_this_iter += 1
                next_hop_map[(o, d)] = new_hop
                
            # 3. ★次の目的地の計算に備え、ベース解（init_next_hop）を即座に随時更新
            for o in PREFECTURES_CODE:
                init_next_hop[(o, d)] = next_hop_map[(o, d)]

        print(f"📊 イテレーション {iter_idx + 1} 終了時の Next-Hop 変更件数: {changes_in_this_iter}")
        
        # 収束判定：1箇所も変更がなければ最適化完了とみなしてブレイク
        if changes_in_this_iter == 0 and not easy_cost_check:
            print("✨ 経路決定が完全に収束しました。反復を終了します。")
            break

    # 6. コンペの提出フォーマット（発地昇順 ➔ 着地昇順 の1次元リスト）に整形
    output_solution = []
    for o in PREFECTURES_CODE:
        for d in PREFECTURES_CODE:
            output_solution.append(next_hop_map[(o, d)])
            
    # 7. 7要素ごとに改行を入れてJSON出力
    os.makedirs(os.path.join(file_dir, method_name), exist_ok=True)
    output_json_path = os.path.join(file_dir, method_name, 'input.json')
    
    json_lines = []
    for i in range(0, len(output_solution), N):
        row_chunk = output_solution[i : i + N]
        row_str = ", ".join(map(str, row_chunk))
        json_lines.append(f"  {row_str}")
    output_string = "[\n" + ",\n".join(json_lines) + "\n]"
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        f.write(output_string)
        
    print(f"正常に最短経路の解を作成しました: {output_json_path}")

if __name__ == '__main__':
    method_name = "dijkstra_new"
    input_case = "dijkstra001"
    evaluate_main("japan", "test", input_case)
    solve_kyushu_backward_dijkstra(method_name,input_case)
    plot_flow_map.main(method_name)
    evaluate_main("japan", "test", method_name)
