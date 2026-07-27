import os
import json
import math
import random
import heapq
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

def main(work_dir, init_answer_folder, opt_answer_folder, max_epochs=100):
    """
    全目的地を順番に走査してツリーを引き直します。
    46拠点すべてを一巡しても1件も改善しなかった場合、完全収束とみなして終了します。
    """
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
    backward_graph = {code: [] for code in prefectures}
    for _, row in df_dist.iterrows():
        u, v = int(row["origin_prefecture_code"]), int(row["destination_prefecture_code"])
        t = float(row["time_min"])
        time_map[(u, v)] = t
        backward_graph[v].append((u, t))

    current_hop = {}
    idx = 0
    for o in prefectures:
        for d in prefectures:
            current_hop[(o, d)] = initial_json_list[idx]
            idx += 1

    current_cost = compute_network_cost(current_hop, flow_data, time_map, prefectures, code_to_idx, N)

    total_improvements = 0
    epoch = 0

    # 2. 周巡型（エポック制）探索ループ
    while epoch < max_epochs:
        epoch += 1
        epoch_improvements = 0
        # print(f"  [Tree Perturb Epoch {epoch}] 46 目的地の網羅的ツリー再生成を開始...")

        # 💡 ランダムではなく、全目的地 d を順番に1つずつ選択
        for d in prefectures:
            # --- A. 現在の全アークの荷量を集計し、積載効率ペナルティを算出 ---
            route_amount = [[0.0] * N for _ in range(N)]
            for (orig, dest), (load, _) in flow_data.items():
                if load <= 0 or orig == dest: continue
                curr = orig
                visited = {curr}
                while curr != dest:
                    nxt = current_hop.get((curr, dest))
                    if nxt is None or nxt in visited: break
                    u_idx, v_idx = code_to_idx[curr], code_to_idx[nxt]
                    route_amount[u_idx][v_idx] += load
                    curr = nxt
                    visited.add(curr)

            # アークごとの仮想コスト（重み）マップを作成
            perturbed_weights = {}
            for v, edges in backward_graph.items():
                for u, t in edges:
                    u_idx, v_idx = code_to_idx[u], code_to_idx[v]
                    L = route_amount[u_idx][v_idx]
                    if L > 0:
                        trucks = max(1, math.ceil(L / 100000.0))
                        eff = L / (trucks * 100000.0)  # 積載率 (0.0～1.0)
                        factor = 1.0 / (eff + 0.05)    # 満車に近いほど安く、スカスカほど高コスト化
                    else:
                        factor = 8.0

                    # 積載率誘導コストに ±15% の微小ノイズを乗せる
                    noise = random.uniform(0.85, 1.15)
                    perturbed_weights[(u, v)] = t * factor * noise

            # --- B. 目的地 d への逆向きダイクストラ法でツリーを一括生成 ---
            dist = {code: float('inf') for code in prefectures}
            parent_node = {code: None for code in prefectures}
            dist[d] = 0.0
            pq = [(0.0, d)]

            while pq:
                cost_v, v = heapq.heappop(pq)
                if cost_v > dist[v]: continue
                for u, _ in backward_graph[v]:
                    w = perturbed_weights[(u, v)]
                    if dist[v] + w < dist[u]:
                        dist[u] = dist[v] + w
                        parent_node[u] = v
                        heapq.heappush(pq, (dist[u], u))

            # バックアップを取りながら、目的地 d 宛てのNext-Hopを一斉上書き
            backups = {}
            for o in prefectures:
                if o == d: continue
                backups[o] = current_hop[(o, d)]
                current_hop[(o, d)] = parent_node[o] if parent_node[o] is not None else d

            # --- C. コスト評価と決定 ---
            new_cost = compute_network_cost(current_hop, flow_data, time_map, prefectures, code_to_idx, N)
            # print(new_cost-current_cost)
            if new_cost < current_cost:
                print(f"    ✨ 改善成功! (着地 {d:2d}): {current_cost:,.2f} -> {new_cost:,.2f}")
                current_cost = new_cost
                epoch_improvements += 1
                total_improvements += 1
            else:
                # 悪化したら即座に復元
                for o, old_v in backups.items():
                    current_hop[(o, d)] = old_v
        if epoch % 50 == 0 or epoch_improvements != 0:
            print(f"  [Tree Perturb Epoch {epoch}] 終了 - この周の改善件数: {epoch_improvements}件")

        # 💡【収束判定】46 拠点すべてを一巡しても1件も改善しなかった場合、完全収束と判定して終了
        # if epoch_improvements == 0:
        #     print("  -> 46 都道府県すべての目的地ツリーの引き直しを試行しましたが、改善が得られなかったため完全収束と判定しました。")
        #     break

    # 改善があった場合のみ上書き保存
    if total_improvements > 0:
        optimized_json_list = []
        for o in prefectures:
            for d in prefectures:
                optimized_json_list.append(current_hop[(o, d)])
        output_path = os.path.join(work_dir, opt_answer_folder, "input.json")
        with open(output_path, "w") as f:
            json.dump(optimized_json_list, f, default=int)

    return total_improvements  # 💡【重要】mixファイルへ改善総数を返す

if __name__ == "__main__":
    work_dir = "case/japan"  # 例: "case/japan" や "kyuusyuu" など
    init_answer_folder = "local_minima_mix"
    opt_answer_folder = "local_minima_redijk"
    main(work_dir,init_answer_folder,opt_answer_folder)
