from pathlib import Path

class PathSetting:

    def __init__(self, root:str, case_name:str, method_name:str = None):
        self.root: Path =Path(root)
        if not self.root.exists():
            raise NotImplementedError(f"{root}はありません。")
        self.case_name:str =case_name
        self.method_name = method_name

    @property
    def case_root(self)->Path:
        return self.root /"case"

    @property
    def case(self) ->Path:
        return self.case_root / self.case_name
    
    @property
    def method(self) ->Path:
        return self.case / self.method_name

    @property
    def network(self)->Path:
        return self.case / "network.json"

    @property
    def order(self)->Path:
        return self.case / "order.csv"

    @property
    def routing_table(self)->Path:
        if self.method_name == None:
            return self.case / "input.json"
        else:
            return self.method / "input.json"
        
    @property
    def save(self)-> Path:
        if self.method_name == None:
            return self.case
        else:
            return self.method

    @property
    def prefectures(self) -> Path:
        return self.case / "prefectures.csv"

    @property
    def truck_distance_time_long(self) -> Path:
        return self.case / "truck_distance_time_long.csv"

    @property
    def virtual_prefecture_flows(self) ->Path:
        return self.case / "virtual_prefecture_flows.csv"
