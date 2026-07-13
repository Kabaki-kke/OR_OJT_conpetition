import os
import sys
import json
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import plot_flow_map

# 💡【分離】アルゴリズム関連は全て solution_generator からインポート
from solution_generator import (
    calculate_route_flows,
    generate_initial_solution,
    generate_new_solution
)

file_dir = Path(__file__).resolve().parent
linehaul_folder_path = os.path.join(file_dir.parent.parent, 'linehaul-problem')
if linehaul_folder_path not in sys.path:
    sys.path.append(linehaul_folder_path)

try:
    from main import main as evaluate_main
except ImportError:
    print("エラー: linehaul-problem/main.py から main 関数をインポートできませんでした。パスを確認してください。")
    sys.exit(1)

def solve_backward_dijkstra(method_name, input_case, max_iters = 200, no_improvement_limit = 10):
    route_file_path = os.path.join(file_dir, 'truck_distance_time_long.csv')
    prefectures_file_path = os.path.join(file_dir, 'prefectures.csv')
    flow_file_path = os.path.join(file_dir, 'virtual_prefecture_flows.csv')
    
    if not os.path.exists(route_file_path):
        print(f"エラー: {route_file_path} が見つかりません。")
        return

    # --- データ構造の準備 ---
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
    # ステップ 1: 初期解の読み込み（または自動生成）
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
        # 💡 アルゴリズム側に初期解生成を依頼
        init_next_hop = generate_initial_solution(PREFECTURES_CODE, backward_graph)

    # ====================================================================
    # ステップ 2: 初期解に基づいて流量を集計
    # ====================================================================
    # 💡 アルゴリズム側に流量集計を依頼
    route_amount, route_variance = calculate_route_flows(init_next_hop, df_flow, code_to_idx, N)

    print(f"🚀 反復最適化を開始します（最大 {max_iters} 周）")
    best_objective = 1e10
    best_next_hop = init_next_hop.copy()
    output_temp_json_path = os.path.join(file_dir, method_name, "temp", 'input.json')
    output_best_json_path = os.path.join(file_dir, method_name, "best", 'input.json')
    
    no_improvement_count = 0  
    original_next_hop = init_next_hop.copy()
            
    for iter_idx in range(max_iters):
        print(f"\n🔄 --- アウター・イテレーション {iter_idx + 1} / {max_iters} ---")
        
        # 💡 アルゴリズム側に新解の生成を依頼
        current_next_hop_map, changes_in_this_iter = generate_new_solution(
            init_next_hop=init_next_hop,
            PREFECTURES_CODE=PREFECTURES_CODE,
            code_to_idx=code_to_idx,
            N=N,
            flow_map=flow_map,
            sd_map=sd_map,
            backward_graph=backward_graph,
            base_time=base_time,
            route_amount=route_amount,
            route_variance=route_variance
        )

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
        
        os.makedirs(os.path.dirname(output_temp_json_path), exist_ok=True)
        with open(output_temp_json_path, 'w', encoding='utf-8') as f:
            f.write(output_string)

        # 評価と最良解の更新判定
        objective = evaluate_main("japan", "test", f"{method_name}/temp", False)
        if objective < best_objective:
            best_objective = objective
            best_next_hop = current_next_hop_map.copy()  
            
            os.makedirs(os.path.dirname(output_best_json_path), exist_ok=True)
            with open(output_best_json_path, 'w', encoding='utf-8') as f:
                f.write(output_string)
            print(f"✨ 最良スコア更新: {objective}")
            init_next_hop = current_next_hop_map.copy()
            no_improvement_count = 0  
        else:
            no_improvement_count += 1
            print(f"⚠️ スコア改善なし ({no_improvement_count}/{no_improvement_limit})")

        # 停滞時のリセット処理
        if no_improvement_count >= no_improvement_limit:
            print("🔄 停滞したため、初期解および初期流量にリセットします。")
            init_next_hop = original_next_hop.copy()
            route_amount, route_variance = calculate_route_flows(init_next_hop, df_flow, code_to_idx, N)
            no_improvement_count = 0

    # ====================================================================
    # ステップ 3: 最終出力
    # ====================================================================
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
    method_name = "iter_01"
    input_case = "dijkstra001"
    max_iter = 100
    ### 解の更新でrandomを用いる場合、適当なステップごとにリセットして大域解を探る
    ### そうでないなら、no_improvement_limit = max_iterとする
    no_improvement_limit = 100
    init_objective = evaluate_main("japan", "test", input_case, False)
    solve_backward_dijkstra(method_name, input_case, max_iter, no_improvement_limit)
    plot_flow_map.main(method_name)
    opt_objective = evaluate_main("japan", "test", method_name, False)
    print(f"初期解：{init_objective}")
    print(f"最適解：{opt_objective}")
