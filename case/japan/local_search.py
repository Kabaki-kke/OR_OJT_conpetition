import os
import json
import math
import random
import pandas as pd

# ==========================================
# 設定: 最適化を行いたいケースのフォルダを指定してください
# ==========================================

def calc_expected_trucks(mu, sigma):
    """
    合計平均(mu)と合計標準偏差(sigma)から、正規分布における切り上げ車両数の期待値を計算[cite: 2]
    """
    if mu <= 0:
        return 0.0
    if sigma <= 0.01:
        # 標準偏差がほぼ0（または単一の確定値）の場合は通常の切り上げ[cite: 2]
        return float(max(1, math.ceil(mu / 100000.0)))
    
    # 誤差関数(math.erf)を利用した正規分布の累積分布関数(CDF)
    def normal_cdf(x, m, s):
        return 0.5 * (1.0 + math.erf((x - m) / (s * 1.4142135623730951)))
    
    # 非負の整数値をとる確率変数の期待値公式: E[Y] = sum_{k=0}^{inf} P(Y > k) を適用
    # Y = ceil(X / 100000) とおくと、Y > k <=> X > k * 100000 となる
    total_expected = 0.0
    k = 0
    while True:
        limit = k * 100000.0
        # P(X > limit) = 1.0 - CDF(limit)
        p_greater = 1.0 - normal_cdf(limit, mu, sigma)
        
        # 確率が極めて小さくなり、かつk*100000が平均を十分に超えたら打ち切り
        if p_greater < 1e-7 and limit > mu:
            break
            
        total_expected += p_greater
        k += 1
        if k > 10000:  # 無限ループ防止の安全弁
            break
            
    return max(1.0, total_expected)

def compute_network_cost(next_hop_map, flow_data, time_map, prefectures, code_to_idx, N):
    """
    現在のNext-Hopマップから、平均・分散の積算を行い正確なトータル期待コストを計算[cite: 2]
    """
    route_amount = [[0.0] * N for _ in range(N)]
    route_variance = [[0.0] * N for _ in range(N)]
    
    # 各ODペアの流量（平均）と分散（標準偏差の二乗）をNext-Hopに従ってアークごとに足し込む[cite: 2]
    for (orig, dest), (load, variance) in flow_data.items():
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
            route_variance[u_idx][v_idx] += variance
            
            curr = nxt
            if curr in visited:  # ループ発生
                is_feasible = False
                break
            visited.add(curr)
            
        if not is_feasible:
            return float('inf')  # 制約違反の解には巨大なペナルティ[cite: 2]

    # トータル期待コストの計算
    total_cost = 0.0
    for u_idx, u in enumerate(prefectures):
        for v_idx, v in enumerate(prefectures):
            mu = route_amount[u_idx][v_idx]
            if mu > 0:
                var = route_variance[u_idx][v_idx]
                sigma = math.sqrt(var)  # 分散の合計の平方根から、この区間のブレ（標準偏差）を算出
                
                trucks = calc_expected_trucks(mu, sigma)
                time_min = time_map.get((u, v), 0.0)
                total_cost += trucks * (10000.0 * time_min / 60.0)  # 車両台数期待値 × 区間費用[cite: 2]
                
    return total_cost

def check_loop_and_reachability(o, d, v, next_hop_map):
    """
    oの次拠点をvに変えたとき、ループせず最終着地dに到達できるか検証
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
    # 1. データの読み込み
    df_pref = pd.read_csv(os.path.join(work_dir, "prefectures.csv"))
    df_flow = pd.read_csv(os.path.join(work_dir, "virtual_prefecture_flows.csv"))
    df_dist = pd.read_csv(os.path.join(work_dir, "truck_distance_time_long.csv"))
    
    with open(os.path.join(work_dir, init_answer_folder, "input.json"), "r") as f:
        initial_json_list = json.load(f)

    # 都道府県コードのソート（解のインデックス順を合わせるため必須）
    prefectures = sorted(df_pref["prefecture_code"].unique())
    N = len(prefectures)
    code_to_idx = {code: i for i, code in enumerate(prefectures)}
    
    # データ構造の整理
    flow_data = {}
    for _, row in df_flow.iterrows():
        flow_data[(int(row["origin_prefecture_code"]), int(row["destination_prefecture_code"]))] = (float(row["virtual_load"]), float(row["standard_deviation"]))
        
    time_map = {}
    adj_nodes = {code: [] for code in prefectures}
    for _, row in df_dist.iterrows():
        u, v = int(row["origin_prefecture_code"]), int(row["destination_prefecture_code"])
        time_map[(u, v)] = float(row["time_min"])
        if v not in adj_nodes[u]:
            adj_nodes[u].append(v)  # 実在アークのみを近傍候補にする

    # 2. 初期解のデコード (1次元リスト -> 辞書マップ)
    current_hop = {}
    idx = 0
    for o in prefectures:
        for d in prefectures:
            current_hop[(o, d)] = initial_json_list[idx]
            idx += 1

    # 初期コストの計算
    current_cost = compute_network_cost(current_hop, flow_data, time_map, prefectures, code_to_idx, N)
    print(f"初期解のコスト: {current_cost:,.2f}")
    if current_cost == float('inf'):
        print("[ERROR] 初期解にループまたは到達不能が含まれています。中断します。")
        return

    # 3. 局所探索（山登り）メインループ
    improvements = 0
    for step in range(1, max_iter + 1):
        o = random.choice(prefectures)
        d = random.choice(prefectures)
        if o == d: continue
        
        old_v = current_hop[(o, d)]
        if not adj_nodes[o]: continue
        
        # 物理的に移動可能な実在アークから、新しいNext-Hop候補をランダム選択
        new_v = random.choice(adj_nodes[o])
        if new_v == old_v: continue
        
        # 仮変更して制約チェック
        current_hop[(o, d)] = new_v
        if check_loop_and_reachability(o, d, new_v, current_hop):
            new_cost = compute_network_cost(current_hop, flow_data, time_map, prefectures, code_to_idx, N)
            
            # 【貪欲判定】コストが厳密に下がった場合のみ解を確定
            if new_cost < current_cost:
                print(f"[Step {step:5d}] コスト改善: {current_cost:,.2f} -> {new_cost:,.2f}")
                current_cost = new_cost
                improvements += 1
                continue
                
        # 悪化または制約違反なら即座にロールバック
        current_hop[(o, d)] = old_v

        if step % 2000 == 0:
            print(f"探索進行中... {step}/{max_iter} 試行完了 (現在のコスト: {current_cost:,.2f})")

    # 4. 最適化解のエンコードと保存 (辞書マップ -> 1次元リスト)
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

if __name__ == "__main__":
    work_dir = "case/japan"  # 例: "case/test" や "kyuusyuu" など
    init_answer_folder = "local_minima"
    opt_answer_folder = "local_minima"
    max_iter = 20000    # 探索を試行する総回数
    main(work_dir, init_answer_folder, opt_answer_folder, max_iter)
