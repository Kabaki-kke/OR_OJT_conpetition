import os
import sys
import json
import heapq
import math
import random
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import plot_flow_map

file_dir = Path(__file__).resolve().parent
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
    required_capacity = load
    return max(1, required_capacity / 100000.0)

def calc_required_trucks(load, variance):
    if load <= 0:
        return 0
    required_capacity = load
    return max(1, math.ceil(required_capacity / 100000.0))

# 💡【追加】解に対応する流量（route_amount）を完全に計算し直すための共通関数
def calculate_route_flows(next_hop_dict, df_flow, code_to_idx, N):
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
            nxt = next_hop_dict[(curr, dest)]
            u_idx, v_idx = code_to_idx[curr], code_to_idx[nxt]
            route_amount[u_idx][v_idx] += load
            route_variance[u_idx][v_idx] += (sd ** 2)
            curr = nxt
            if curr in visited: break
            visited.add(curr)
    return route_amount, route_variance

def solve_backward_dijkstra(method_name, input_case, max_iters = 200, no_improvement_limit = 10):
    route_file_path = os.path.join(file_dir, 'truck_distance_time_long.csv')
    prefectures_file_path = os.path.join(file_dir, 'prefectures.csv')
    flow_file_path = os.path.join(file_dir, 'virtual_prefecture_flows.csv')
    
    if not os.path.exists(route_file_path):
        print(f"エラー: {route_file_path} が見つかりません。")
        return

    df_route = pd.read_csv(route_file_path)
    df_pref = pd.read_csv(prefectures_file_path)
    df_flow = pd.read_csv(flow_file_path)
    PREFECTURES_CODE = sorted(list(df_pref["prefecture_code"]))

    flow_map = df_flow.set_index(['origin_prefecture_code', 'destination_prefecture_code'])['virtual_load'].to_dict()
    sd_map = df_flow.set_index(['origin_prefecture_code', 'destination_prefecture_code'])['standard_deviation'].to_dict()

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
    # ステップ 1: 初期解の読み込み
    # ====================================================================
    init_next_hop = {}
    initial_json_path = os.path.join(file_dir, input_case, 'input.json') 
    
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
        print(f"⚠️ 最短パスで初期解を自動生成します。")
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
    # ステップ 2: 初期解に基づいて流量を集計
    # ====================================================================
    # 💡関数化してスッキリさせました
    route_amount, route_variance = calculate_route_flows(init_next_hop, df_flow, code_to_idx, N)

    print(f"🚀 反復最適化を開始します（最大 {max_iters} 周）")
    best_objective = 1e10
    best_next_hop = init_next_hop.copy()  # 💡【修正】初期解を暫定最良として保持
    output_temp_json_path = os.path.join(file_dir, method_name, "temp", 'input.json')
    output_best_json_path = os.path.join(file_dir, method_name, "best", 'input.json')
    
    no_improvement_count = 0  # 💡【修正】スコアが更新されない回数をカウントするよう変更
    original_next_hop = init_next_hop.copy()
            
    for iter_idx in range(max_iters):
        changes_in_this_iter = 0
        next_hop_map = init_next_hop.copy()
        current_next_hop_map = init_next_hop.copy()
        calculator = calc_required_trucks_for_early_iteration
 

        print(f"\n🔄 --- アウター・イテレーション {iter_idx + 1} / {max_iters} ---")
        
        d_list = PREFECTURES_CODE.copy()
        random.shuffle(d_list)
        
        for d in d_list:
            # --- 🔄 随時更新【A】: 古い流量を一旦マイナス ---
            for o in PREFECTURES_CODE:
                if o == d or o not in code_to_idx or d not in code_to_idx: continue
                load = flow_map.get((o, d), 0.0)
                sd = sd_map.get((o, d), 0.0)
                if load <= 0: continue
                
                curr = o
                visited = {curr}
                while curr != d:
                    nxt = current_next_hop_map[(curr, d)]
                    u_idx, v_idx = code_to_idx[curr], code_to_idx[nxt]
                    route_amount[u_idx][v_idx] = max(0.0, route_amount[u_idx][v_idx] - load)
                    route_variance[u_idx][v_idx] = max(0.0, route_variance[u_idx][v_idx] - (sd ** 2))
                    curr = nxt
                    if curr in visited: break
                    visited.add(curr)

            # --- 🔍 本番ダイクストラ法 ---
            cost = {code_from: {code_curr: float('inf') for code_curr in PREFECTURES_CODE} for code_from in PREFECTURES_CODE}
            parent_node = {code: None for code in PREFECTURES_CODE}
            cost[d][d] = 0
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
                    total_incremental_cost = (ta_uv - tb_uv) * (10000.0 * time / 60.0) + time * 10
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
                        
            for o in PREFECTURES_CODE:
                old_hop = current_next_hop_map[(o, d)]
                new_hop = parent_node[o] if parent_node[o] is not None else d
                if o == d: new_hop = o
                
                if old_hop != new_hop:
                    changes_in_this_iter += 1
                next_hop_map[(o, d)] = new_hop
                
            for o in PREFECTURES_CODE:
                current_next_hop_map[(o, d)] = next_hop_map[(o, d)]

        print(f"📊 イテレーション {iter_idx + 1} 終了時の Next-Hop 変更件数: {changes_in_this_iter}")
        
        # tempに保存するための整形
        output_solution = []
        for o in PREFECTURES_CODE:
            for d in PREFECTURES_CODE:
                output_solution.append(current_next_hop_map[(o, d)])
        
        json_lines = []
        for i in range(0, len(output_solution), N):
            row_chunk = output_solution[i : i + N]
            row_str = ", ".join(map(str, row_chunk))
            json_lines.append(f"  {row_str}")
        output_string = "[\n" + ",\n".join(json_lines) + "\n]"
        
        with open(output_temp_json_path, 'w', encoding='utf-8') as f:
            f.write(output_string)

        # 評価と最良解の更新判定
        objective = evaluate_main("kyuusyuu", "test", "dijkstra_iter/temp", False)
        if objective < best_objective:
            best_objective = objective
            best_next_hop = current_next_hop_map.copy()  # 最良解のマップを保持
            
            with open(output_best_json_path, 'w', encoding='utf-8') as f:
                f.write(output_string)
            print(f"✨ 最良スコア更新: {objective}")
            init_next_hop = next_hop_map.copy()
            no_improvement_count = 0  # スコアが更新されたのでリセット
        else:
            no_improvement_count += 1
            print(f"⚠️ スコア改善なし ({no_improvement_count}/5)")

        # 💡【修正】スコアが5回連続で改善しなかったらリセットを実行
        if no_improvement_count >= no_improvement_limit:
            print("🔄 停滞したため、初期解および初期流量にリセットします。")
            init_next_hop = original_next_hop.copy()
            # 💡【最重要】流量（route_amount）も初期状態の綺麗な値にリセットする！
            route_amount, route_variance = calculate_route_flows(init_next_hop, df_flow, code_to_idx, N)
            no_improvement_count = 0


    # ====================================================================
    # ステップ 3: 最終出力
    # ====================================================================
    # 💡【修正】出力する解を `next_hop_map` ではなく、記録しておいた `best_next_hop` に変更
    output_solution = []
    for o in PREFECTURES_CODE:
        for d in PREFECTURES_CODE:
            output_solution.append(best_next_hop[(o, d)])
    
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
    method_name = "dijkstra_iter"
    input_case = "dijkstra001"
    max_iter = 200
    no_improvement_limit = 10
    evaluate_main("kyuusyuu", "test", input_case)
    solve_backward_dijkstra(method_name, input_case, max_iter, no_improvement_limit)
    plot_flow_map.main(method_name)
    evaluate_main("kyuusyuu", "test", method_name)
