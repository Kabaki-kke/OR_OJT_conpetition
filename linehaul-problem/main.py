"""Example problem."""

import json
import logging
from pathlib import Path
import sys

import click

from evaluator.main import evaluate
from infra.path_setting import PathSetting
from app.problem_factory import ProblemFactory
from validators.example_env import validate_example_env
from validators.variable import validate_variable

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)  # Logger


# @click.command(help="Example problem.")
# @click.option(
#     "-c",
#     "--case_name",
#     type=str,
#     help="case_name",
# )
# @click.option(
#     "-t",
#     "--case_type",
#     type=str,
#     help="case_type",
# )
def main(case_name: str, case_type) -> None:
    """Example main function for evaluating given variable."""
    # このファイルの親の親の場所　/problem-linehaul
    root = Path(__file__).parent.parent
    env_var = {"case_name": case_name,
               "case_type": case_type}

    try:
        validated_example_env = validate_example_env(env_var)  # Validate the environment variable
        msg = f"{validated_example_env}"
        LOGGER.info(msg)
        # case_nameをもとに、諸々のファイルがある場所を指定
        path_setting = PathSetting(str(root), validated_example_env["case_name"].strip())
        
        problem = ProblemFactory.create(validated_example_env["case_type"].strip(),
                                        path_setting)

        problem.validate()  # Validate the variable
        # msg = f"validated_variable: {validated_variable}"
        # LOGGER.info(msg)

        result = evaluate(problem, path_setting)  # Evaluate the variable
        msg = f"result: {result}"
        LOGGER.info(msg)

        sys.stdout.write(json.dumps(result) + "\n")  # Write the result to the standard output

    except Exception as e:
        error_result = {
            "objective": None,
            "feasible": False,
            "constraint": None,
            "error": str(e),
        }
        sys.stdout.write(
            json.dumps(error_result, ensure_ascii=False) + "\n",
        )  # Write the error result to the standard output
        msg = f"error result: {error_result}"
        LOGGER.info(msg)


if __name__ == "__main__":
    # case_type = "test"　　なら　/problem-linehaul/case/{case_name}/input.json　を読み込む
    # case_type = "problem" なら　手入力
    main("kyuusyuu", "test")
