from app.test_problem.problem import Problem
from infra.path_setting import PathSetting
from app.real_problem.real_varidator import RealVariableValidator


class RealProblem(Problem):

    VaridatorClass = RealVariableValidator

    def initialize(self, path_setting: PathSetting):
            self.evaluator.initialize(path_setting)
            # 総ノード数から都道府県数を逆算 (49なら7、2116なら46)
            n_prefecture = round(self.n_node ** 0.5)
            variable_2d_array =[self.variable[i:i+n_prefecture]
                                for i in range(0,len(self.variable),n_prefecture)]
            self.evaluator.set_route_table(variable_2d_array)
