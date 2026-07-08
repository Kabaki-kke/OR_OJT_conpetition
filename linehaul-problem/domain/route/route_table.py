
from domain.network.network_entity import NetWork
from domain.node.node_data import NodeData
from domain.branch.branch_collection import BranchCollection
from domain.node.node_collection import NodeCollection
from .route_entity import RouteEntity
from ..order.order_vo import OrderVO


class RouteTable:

    def __init__(self,
                 network: NetWork):
        self.network = network
        self.next_hop_table: dict[tuple[NodeData, NodeData], NodeData] = dict()

    @property
    def nodes(self) -> NodeCollection:
        return self.network.node

    @property
    def branches(self) -> BranchCollection:
        return self.network.branch

    @property
    def route_node_pairs(self):
        return self.next_hop_table.keys()

    def set_list(self, data: list):
        for d in data:
            from_ = self.nodes.data_dict.get(d["from"], None)
            if from_ is None:
                raise ValueError(f"{from_}はノードリストに登録されていません")
            nexts = d["next"]
            for to_, _next in zip(self.nodes.data_list, nexts):
                if from_ == to_:
                    continue
                __next = self.nodes.data_dict.get(_next, None)
                if __next is None:
                    raise ValueError(f"{_next}はノードリストに登録されていません")
                if from_ == __next:
                    raise ValueError(f"{from_}のnext_hopに{from_}が指定されています")
                if not self.branches.is_connected(from_, __next):
                    raise ValueError(f"{from_}と{__next}は接続していません")

                self.next_hop_table[(from_, to_)] = __next

    def set_route_list_by_1Dlist(self, data:list[list]):
            for i, d in enumerate(data):
                # 1から始まる連番ではなく、読み込まれたノードリストのi番目を直接取得する
                if i >= len(self.nodes.data_list):
                    raise ValueError(f"データの行数がノードリストの数（{len(self.nodes.data_list)}）を超えています")
                from_ = self.nodes.data_list[i]
                
                for to_, _next in zip(self.nodes.data_list, d):
                    if from_ == to_:
                        continue
                    __next = self.nodes.data_dict.get(_next, None)
                    if __next is None:
                        raise ValueError(f"{_next}はノードリストに登録されていません")
                    if from_ == __next:
                        raise ValueError(f"{from_}のnext_hopに{from_}が指定されています")
                    if not self.branches.is_connected(from_, __next):
                        raise ValueError(f"{from_}と{__next}は接続していません")

                    self.next_hop_table[(from_, to_)] = __next


    def get_next(self, from_: NodeData, to_: NodeData) -> NodeData:
        return self.next_hop_table[(from_, to_)]

    def get_route_by_order(self, order: OrderVO) -> RouteEntity:
        return self.get_route(order.from_node, order.to_node)

    def get_route(self, from_: NodeData, to_: NodeData) -> RouteEntity:
        if from_ == to_:
            raise ValueError(f"fromとtoが同じです。{from_}")

        route = [from_]
        visited = {from_}
        curr = from_

        while curr != to_:
            next_ = self.get_next(curr, to_)
            route.append(next_)

            if next_ in visited:
                raise ValueError(f"{from_}から{to_}へのルートに循環があります。{route}")
            visited.add(next_)
            if next_ == to_:
                return RouteEntity(route)
            curr = next_
        raise ValueError(f"{from_}から{to_}へのルートが存在しません。")
