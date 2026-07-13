import heapq
import math
import random

# アルゴリズム内部の履歴を保持するためのモジュール変数
_arc_weights = {}
_last_next_hop = None

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

def calculate_route_flows(next_hop_dict, df_flow, code_to_idx, N):
    """
    【メイン用】指定されたNext-Hopに基づいて、区間ごとの流量（平均・分散）を計算する
    """
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

def _calculate_route_flows_from_maps(next_hop_dict, flow_map, sd_map, code_to_idx, N):
    """
    【内部用】DataFrameを介さず、マッピング辞書から高速に流量を再集計する
    """
    route_amount = [[0.0] * N for _ in range(N)]
    route_variance = [[0.0] * N for _ in range(N)]
    for (orig, dest), load in flow_map.items():
        sd = sd_map.get((orig, dest), 0.0)
        if load <= 0: continue
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

def generate_initial_solution(PREFECTURES_CODE, backward_graph):
    """
    初期解が存在しない場合、ダイクストラ法により最短経路の初期解を生成する
    """
    init_next_hop = {}
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
    return init_next_hop

def generate_new_solution(
    init_next_hop,
    PREFECTURES_CODE,
    code_to_idx,
    N,
    flow_map,
    sd_map,
    backward_graph,
    base_time,
    route_amount,
    route_variance
):

    global _arc_weights, _last_next_hop
    
    # 💡 メイン側での停滞リセット（または初回実行）が入った場合、重みの履歴をクリアして同期する
    if _last_next_hop is not None and init_next_hop != _last_next_hop:
        _arc_weights.clear()

    # 1. 前回の流量集計を元に、各アークの「通過単位コスト（重み）」を算出
    alpha = 0.1  # スムージング係数（小さいほどゆっくり変化し、あっちこっちへのルートの激しい振動を防ぐ）
    new_weights = {}
    
    
    for v, edges in backward_graph.items():
        for u, time in edges:
            u_idx, v_idx = code_to_idx[u], code_to_idx[v]
            L = route_amount[u_idx][v_idx]
            V = route_variance[u_idx][v_idx]
            
            if L > 0:
                T = calc_required_trucks(L, V)
                # トラック1台（10万kg満載）あたりのコスト効率を計算
                # 荷物が少なく、積載率が悪いアークほどペナルティ（eff）が大きくなる
                eff = T / (L / 100000.0)
                eff = max(1.0, min(15.0, eff))  # 極端なペナルティを防ぐクリッピング
            else:
                # 流量ゼロのアークは、他のアークへの集約を促すために高めのペナルティを課す
                eff = 15.0
                
            new_weights[(u, v)] = time * eff

    # 重みの履歴を更新（指数平滑化によるマイルドなブレンド）
    if not _arc_weights:
        _arc_weights = new_weights
    else:
        for k in new_weights:
            _arc_weights[k] = alpha * new_weights[k] + (1.0 - alpha) * _arc_weights.get(k, new_weights[k])

    # 2. 更新された固定重みマップを元に、全目的地に対して一斉に最短経路を探索
    next_hop_map = {}
    changes_in_this_iter = 0
    
    for d in PREFECTURES_CODE:
        dist = {code: float('inf') for code in PREFECTURES_CODE}
        parent_node = {code: None for code in PREFECTURES_CODE}
        dist[d] = 0
        pq = [(0.0, d)]
        
        while pq:
            current_cost, v = heapq.heappop(pq)
            if current_cost > dist[v]: continue
            
            for u, _ in backward_graph[v]:
                # 算出したアーク重みを取得（なければベース時間*15）
                weight = _arc_weights.get((u, v), base_time[(u, v)] * 15.0)
                if dist[v] + weight < dist[u]:
                    dist[u] = dist[v] + weight
                    parent_node[u] = v
                    heapq.heappush(pq, (dist[u], u))
                    
        # マップの構築と変更件数のカウント
        for o in PREFECTURES_CODE:
            if o == d:
                new_hop = o
            else:
                new_hop = parent_node[o] if parent_node[o] is not None else d
                
            old_hop = init_next_hop.get((o, d))
            if old_hop != new_hop:
                changes_in_this_iter += 1
            next_hop_map[(o, d)] = new_hop

    # 3. メイン側へ同期を引き継ぐため、新しい解に対応する正確な流量を破壊的に再計算して上書き
    new_amount, new_variance = _calculate_route_flows_from_maps(next_hop_map, flow_map, sd_map, code_to_idx, N)
    for i in range(N):
        for j in range(N):
            route_amount[i][j] = new_amount[i][j]
            route_variance[i][j] = new_variance[i][j]
            
    # 状態の記録
    _last_next_hop = next_hop_map.copy()

    return next_hop_map, changes_in_this_iter
