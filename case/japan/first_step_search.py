import os
import json
import math
import random
from statistics import NormalDist
import pandas as pd

def compute_network_cost(next_hop_map, flow_data, time_map, prefectures, code_to_idx, N):
    """公式完全互換の高精度コスト計算関数"""
    route_amount = [[0.0] * N for _ in range(N)]
    route_variance = [[0.0] * N for _ in range(N)]
    
    for (orig, dest), (load, sd) in flow_data.items():
        if load <= 0 or orig == dest: 
            continue
        curr = orig
        visited = {curr}
        is_feasible = True
        while curr != dest:
            nxt = next_hop_map.get((curr, dest))
            if nxt is None or curr not in code_to_idx or nxt not in code_to_idx:
                is_feasible = False
                break
            u_idx, v_idx = code_to_idx[curr], code_to_idx[nxt]
            route_amount[u_idx][v_idx] += load
            route_variance[u_idx][v_idx] += sd ** 2
            curr = nxt
            if curr in visited:
                is_feasible = False
                break
            visited.add(curr)
        if not is_feasible:
            return float('inf')

    total_cost = 0.0
    EPSILON = 10 ** -10
    stop_prob = 1.0 - EPSILON
    VEHICLE_CAPACITY = 100000.0
    
    for u_idx, u in enumerate(prefectures):
        for v_idx, v in enumerate(prefectures):
            mu = route_amount[u_idx][v_idx]
            if mu > 0:
                var = route_variance[u_idx][v_idx]
                sigma = math.sqrt(var)
                time_min = time_map.get((u, v), 0.0)
                branch_cost = 80000.0 * time_min / 480.0
                norm_dist = NormalDist(mu=mu, sigma=sigma if sigma > 0 else 1e-9)
                cdf_list = [0.0]
                n_vehicle = 1
                while cdf_list[-1] < stop_prob:
                    prob = norm_dist.cdf(n_vehicle * VEHICLE_CAPACITY)
                    cdf_list.append(prob)
                    n_vehicle += 1
                edge_cost = 0.0
                for n in range(len(cdf_list)):
                    upper = cdf_list[n]
                    lower = cdf_list[n - 1] if n > 1 else 0.0
                    edge_cost += (upper - lower) * n * branch_cost
                total_cost += edge_cost
    return total_cost

def check_loop_and_reachability(o, d, v, next_hop_map):
    """ループおよび到達可能性の検証"""
    curr = v
    visited = {o, curr}
    while curr != d:
        nxt = next_hop_map.get((curr, d))
        if nxt is None or nxt in visited: 
            return False
        visited.add(nxt)
        curr = nxt
    return True

def main(work_dir, init_answer_folder, opt_answer_folder, max_iter):
    # 1. データの読み込み
    df_pref = pd.read_csv(os.path.join(work_dir, "prefectures.csv"))
    df_flow = pd.read_csv(os.path.join(work_dir, "virtual_prefecture_flows.csv"))
    df_dist = pd.read_csv(os.path.join(work_dir, "truck_distance_time_long.csv"))
    
    target_json_path = os.path.join(work_dir, init_answer_folder, "input.json")
    with open(target_json_path, "r") as f:
        initial_json_list = json.load(f)

    prefectures = sorted(df_pref["prefecture_code"].unique())
    N = len(prefectures)
    code_to_idx = {code: i for i, code in enumerate(prefectures)}
    
    flow_data = {}
    for _, row in df_flow.iterrows():
        flow_data[(int(row["origin_prefecture_code"]), int(row["destination_prefecture_code"]))] = (float(row["virtual_load"]), float(row["standard_deviation"]))
        
    time_map = {}
    adj_nodes = {code: [] for code in prefectures}
    all_real_edges = []  # 存在する物理アークのリスト[cite: 6]
    for _, row in df_dist.iterrows():
        u, v = int(row["origin_prefecture_code"]), int(row["destination_prefecture_code"])
        time_map[(u, v)] = float(row["time_min"])
        if v not in adj_nodes[u]:
            adj_nodes[u].append(v)
        all_real_edges.append((u, v))

    current_hop = {}
    idx = 0
    for o in prefectures:
        for d in prefectures:
            current_hop[(o, d)] = initial_json_list[idx]
            idx += 1

    current_cost = compute_network_cost(current_hop, flow_data, time_map, prefectures, code_to_idx, N)

    # 2. 探索（ワーストアーク破壊・一斉迂回）ループ
    improvements = 0
    for step in range(1, max_iter + 1):
        # 物理的に存在する実在アークからランダムに1つのアーク (u, v) を選ぶ[cite: 6]
        u, v = random.choice(all_real_edges)
        
        # 現在のネットワークで、このアーク (u, v) を「最初の一歩」として利用している目的地 d をすべて集める[cite: 6]
        target_d_list = []
        for d in prefectures:
            if u == d: continue
            if current_hop[(u, d)] == v:
                target_d_list.append(d)
                
        # このアークを誰も使っていなければスキップ
        if not target_d_list: continue
        
        # u からの移動先として、現在の v 以外の別の隣接ノード w をランダムに選ぶ[cite: 6]
        if not adj_nodes[u] or len(adj_nodes[u]) <= 1: continue
        w = random.choice([node for code in prefectures for node in adj_nodes[u] if node != v])
        
        # このアークを通過していたすべての目的地行きのNext-Hopを、一斉に w へ迂回させて仮更新する[cite: 5]
        backups = {}
        for d in target_d_list:
            backups[d] = current_hop[(u, d)]
            current_hop[(u, d)] = w
            
        # 一斉変更によってループや到達不能エラーが発生しないか一括チェック[cite: 6]
        is_block_feasible = True
        for d in target_d_list:
            if not check_loop_and_reachability(u, d, w, current_hop):
                is_block_feasible = False
                break
                
        if is_block_feasible:
            # コストの再計算
            new_cost = compute_network_cost(current_hop, flow_data, time_map, prefectures, code_to_idx, N)
            
            # 【貪欲判定】アークを潰して別ルートにまとめた方が、全体の期待車両台数が浮いて安くなった場合のみ確定
            if new_cost < current_cost:
                print(f"[Arc Destruct {step:5d}] 改善成功! {current_cost:,.2f} -> {new_cost:,.2f} (アーク({u}->{v})を通過する {len(target_d_list)}件の目的地ルートを {w} へ一斉迂回)")
                current_cost = new_cost
                improvements += 1
                continue

        # 悪化または制約違反なら塊全体をロールバック[cite: 6]
        for d, old_v in backups.items():
            current_hop[(u, d)] = old_v
            
        if step % 1000 == 0:
            print(f"探索進行中... {step}/{max_iter} 試行完了 (現在のコスト: {current_cost:,.2f})")


    # 3. 改善があった場合のみ上書き保存[cite: 5]
    if improvements > 0:
        optimized_json_list = []
        for o in prefectures:
            for d in prefectures:
                optimized_json_list.append(current_hop[(o, d)])
        output_path = os.path.join(work_dir, opt_answer_folder, "input.json")
        with open(output_path, "w") as f:
            json.dump(optimized_json_list, f)
            
    return improvements  # 💡【重要】mixファイルが検知できるよう、確定件数をreturnする[cite: 5]
