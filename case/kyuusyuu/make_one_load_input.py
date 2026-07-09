import json
import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd

def create_one_line_input(path: list[int], output_filename: str = "input.json"):
    """
    指定された都道府県IDのリスト（path）の順番に一列に幹線道路が繋がる input.json を作成する。
    """
    # 評価器（RouteTable）の読み込み順序に適合させるため、
    # データの出力順（外側・内側ループ）は都道府県IDの昇順にする必要があります。
    all_nodes = sorted(path)
    
    result = []
    
    # 外側ループ：発ノード (from_node)
    for from_node in all_nodes:
        idx_from = path.index(from_node)
        
        # 内側ループ：着ノード (to_node)
        for to_node in all_nodes:
            # 自分自身へのNext-Hop
            # （評価ロジック側で from_node == to_node の時はスキップされるため、
            #  エラー回避用に自分自身のIDを入れておきます）
            if from_node == to_node:
                result.append(from_node)
                continue
                
            idx_to = path.index(to_node)
            
            # パス上での位置関係（インデックス）を比較して Next-Hop（隣のノード）を決定
            if idx_from < idx_to:
                # 目的地がパスの「先（右側）」にある場合は、1つ進む
                next_hop = path[idx_from + 1]
            else:
                # 目的地がパスの「手前（左側）」にある場合は、1つ戻る
                next_hop = path[idx_from - 1]
                
            result.append(next_hop)
            
    # JSONファイルとして書き出し
    file_dir = Path(__file__).resolve().parent
    output_filename = os.path.join(file_dir ,output_filename)
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
        
    print(f"正常に {output_filename} を作成しました。（要素数: {len(result)}）")


if __name__ == "__main__":
    node_path = [42,41,40,43,44,45,46]
    create_one_line_input(node_path)
