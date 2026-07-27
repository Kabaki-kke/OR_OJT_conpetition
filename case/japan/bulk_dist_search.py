import os
import json
import math
import random
from statistics import NormalDist
import pandas as pd

def compute_network_cost(next_hop_map, flow_data, time_map, prefectures, code_to_idx, N):
    """
    現在のNext-Hopマップから、平均・分散の積算を行い、
    公式の離ストレージ・打ち切り仕様に完全準拠して正確なトータル期待コストを計算します[cite: 4, 5]。
    """
    route_amount = [[0.0] * N for _ in range(N)]
    route_variance = [[0.0] * N for _ in range(N)]
    
    # 各ODペアの流量（平均）と分散（標準偏差の二乗）をNext-Hopに従ってアークごとに足し込む[cite: 4, 5]
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
            route_variance[u_idx][v_idx] += sd ** 2  # 標準偏差の二乗（分散）を合算[cite: 4, 5]
            
            curr = nxt
            if curr in visited:  # ループ発生[cite: 4]
                is_feasible = False
                break
            visited.add(curr)
            
        if not is_feasible:
            return float('inf')  # 制約違反の解には巨大なペナルティ[cite: 4]

    # トータル期待コストの計算（公式ロジックの完全再現）[cite: 5]
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
    """
    oの次拠点をvに変えたとき、ループせず最終着地dに到達できるか検証[cite: 4]
    """
    curr = v
    visited = {o, curr}
    while curr != d:
        nxt = next_hop_map.get((curr, d))
        if nxt is None: return False
        if nxt in visited: return False
        visited.add(nxt)
        curr = nxt
    return True

def main(work_dir, init_answer_folder, opt_answer_folder, max_iter):
    # 1. データの読み込み[cite: 4]
    df_pref = pd.read_csv(os.path.join(work_dir, "prefectures.csv"))
    df_flow = pd.read_csv(os.path.join(work_dir, "virtual_prefecture_flows.csv"))
    df_dist = pd.read_csv(os.path.join(work_dir, "truck_distance_time_long.csv"))
    
    with open(os.path.join(work_dir, init_answer_folder, "input.json"), "r") as f:
        initial_json_list = json.load(f)

    prefectures = sorted(df_pref["prefecture_code"].unique())
    N = len(prefectures)
    code_to_idx = {code: i for i, code in enumerate(prefectures)}
    
    flow_data = {}
    for _, row in df_flow.iterrows():
        flow_data[(int(row["origin_prefecture_code"]), int(row["destination_prefecture_code"]))] = (float(row["virtual_load"]), float(row["standard_deviation"]))
        
    time_map = {}
    adj_nodes = {code: [] for code in prefectures}
    for _, row in df_dist.iterrows():
        u, v = int(row["origin_prefecture_code"]), int(row["destination_prefecture_code"])
        time_map[(u, v)] = float(row["time_min"])
        if v not in adj_nodes[u]:
            adj_nodes[u].append(v)  # 実在アークのみを近傍候補にする[cite: 4]

    # 初期解のデコード[cite: 4]
    current_hop = {}
    idx = 0
    for o in prefectures:
        for d in prefectures:
            current_hop[(o, d)] = initial_json_list[idx]
            idx += 1

    current_cost = compute_network_cost(current_hop, flow_data, time_map, prefectures, code_to_idx, N)
    print(f"初期解のコスト: {current_cost:,.2f}")
    if current_cost == float('inf'):
        print("[ERROR] 初期解にループまたは到達不能が含まれています。中断します。")
        return

    # 3. 塊（アーク集約）一斉変更型 局所探索ループ
    improvements = 0
    for step in range(1, max_iter + 1):
        # ① ランダムに都道府県aを選ぶ
        a = random.choice(prefectures)
        if not adj_nodes[a]: continue
        
        # aから出ているNext-Hop先ごとに目的地dをグループ分類する
        dest_by_next = {}
        for d in prefectures:
            if a == d: continue
            old_v = current_hop[(a, d)]
            if old_v not in dest_by_next:
                dest_by_next[old_v] = []
            dest_by_next[old_v].append(d)
            
        if not dest_by_next: continue
        
        # ② aのNext-Hop先を確認し、そのうち一種類bをランダムに選ぶ
        b = random.choice(list(dest_by_next.keys()))
        target_d_list = dest_by_next[b]
        
        # ③ すべてのbを、実在するランダムな変更先c（b以外）に更新する[cite: 4]
        c = random.choice(adj_nodes[a])
        if c == b: continue
        
        # バックアップを取りつつ一斉に仮更新
        backups = {}
        for d in target_d_list:
            backups[d] = current_hop[(a, d)]
            current_hop[(a, d)] = c
            
        # ⑤ すべての変更先でループや到達不能がないか一括制約確認
        is_block_feasible = True
        for d in target_d_list:
            if not check_loop_and_reachability(a, d, c, current_hop):
                is_block_feasible = False
                break
                
        # 複合チェックを通過した場合のみコスト評価へ進む
        if is_block_feasible:
            # ④ スコアを確認する
            new_cost = compute_network_cost(current_hop, flow_data, time_map, prefectures, code_to_idx, N)
            
            if new_cost < current_cost:
                print(f"[Step {step:5d}] 塊一斉変更成功! コスト削減: {current_cost:,.2f} -> {new_cost:,.2f} (拠点 {a} から {b} 向かいのターゲット {len(target_d_list)}件をすべて {c} へ)")
                current_cost = new_cost
                improvements += 1
                continue

        # 悪化、または1件でもループ等の制約違反があれば、塊全体を即座にロールバック
        for d, old_v in backups.items():
            current_hop[(a, d)] = old_v

        if step % 2000 == 0:
            print(f"探索進行中... {step}/{max_iter} 試行完了 (現在のコスト: {current_cost:,.2f})")

    # 4. 最適化解のエンコードと保存[cite: 4]
    optimized_json_list = []
    for o in prefectures:
        for d in prefectures:
            optimized_json_list.append(current_hop[(o, d)])
            
    output_path = os.path.join(work_dir, opt_answer_folder, "input.json")
    with open(output_path, "w") as f:
        json.dump(optimized_json_list, f)
        
    print(f"\n--- 探索完了 ---")
    print(f"総改善件数: {improvements} 件")
    print(f"最終コスト: {current_cost:,.2f}")
    print(f"最適化された解を保存しました: {output_path}")
    return improvements

if __name__ == "__main__":
    work_dir = "case/japan"  # 例: "case/japan" や "kyuusyuu" など
    init_answer_folder = "local_minima_mix"
    opt_answer_folder = "local_minima_mix"
    for i in range(1):
        print(f"計算 {i+1}/4")
        max_iter = 10000    # 探索を試行する総回数
        imp = main(work_dir, init_answer_folder, opt_answer_folder, max_iter)
