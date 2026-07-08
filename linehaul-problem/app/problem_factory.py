
import csv
from pathlib import Path
from infra.path_setting import PathSetting  # 追加

from .real_problem.real_cost_evaluator import RealCostEvaluator
from .real_problem.real_problem import RealProblem
from .test_problem.problem import Problem


class ProblemFactory:
    @staticmethod
    def create(case_type: str, path_setting: PathSetting) -> Problem:  # 引数をPathSettingに変更
        file_path = path_setting.routing_table

        if case_type == "test":
            if not file_path.exists():
                raise FileNotFoundError(f"{file_path} is not exist")

        # prefectures.csv から現在のケースの都道府県数を動的にカウント
        n_pref = 0
        if path_setting.prefectures.exists():
            with open(path_setting.prefectures, mode='r', encoding='utf_8_sig') as f:
                reader = csv.reader(f)
                next(reader)  # ヘッダーをスキップ
                n_pref = sum(1 for row in reader if row)
        else:
            n_pref = 46  # デフォルト値

        n_node_total = n_pref * n_pref  # 1次元配列の総要素数 (7*7=49 や 46*46=2116)

        match case_type:
            case "test":
                variable = file_path.read_text()
                problem = RealProblem(str_variable=variable, evaluator=RealCostEvaluator(v_capa=100000), n_node=n_node_total)

            case "problem":
                variable = input()
                problem = RealProblem(str_variable=variable, evaluator=RealCostEvaluator(v_capa=100000), n_node=n_node_total)
            case _:
                raise ValueError(f"case_type:{case_type} is undifined")
        return problem
