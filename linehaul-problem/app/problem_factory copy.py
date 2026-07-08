from pathlib import Path

from .real_problem.real_cost_evaluator import RealCostEvaluator
from .real_problem.real_problem import RealProblem
from .test_problem.problem import Problem


class ProblemFactory:
    @staticmethod
    def create(case_type: str, file_path: Path) -> Problem:
        if case_type == "test":
            if not file_path.exists():
                raise FileNotFoundError(f"{file_path}is not exist")

        match case_type:
            case "test":
                variable = file_path.read_text()
                problem = RealProblem(str_variable=variable, evaluator=RealCostEvaluator(v_capa=100000), n_node=46 * 46)

            case "problem":
                variable = input()
                problem = RealProblem(str_variable=variable, evaluator=RealCostEvaluator(v_capa=100000), n_node=46 * 46)
            case _:
                raise ValueError(f"case_type:{case_type} is undifined")
        return problem
