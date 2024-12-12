from torch_geometric.data import HeteroData

import torch
import json
import pickle
from bson.binary import Binary


def read_entities(file_path: str, size: int = -1):
    """Read a JSON entity file row by row, parse as dictionary on the fly"""
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if size is not None and size > 0 and i >= size:
                break
            data = json.loads(line)
            yield data


def read_relations(file_path: str, size: int = -1):
    """Read a txt relation file row by row, parse as a pair of IDs on the fly"""
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if size is not None and size > 0 and i >= size:
                break
            obj1, obj2 = line.split()
            yield obj1, obj2


def chunks(arr, n: int):
    """Split array into many equal sized chunks"""
    n = max(1, n)
    return (arr[i : i + n] for i in range(0, len(arr), n))


# def to_binary(tensor: torch.Tensor):
#     return Binary(pickle.dumps(tensor, protocol=2))


# def load_binary(blob: Binary):
#     return pickle.loads(blob)


# def binary_hetero_data(data: HeteroData):
#     x_dict = {}
#     edge_index_dict = {}

#     for k, v in data.x_dict.items():
#         x_dict[k] = to_binary(v)
#     for k, v in data.edge_index_dict.items():
#         edge_index_dict["___".join(k)] = to_binary(v)

#     return {
#         "x_dict": x_dict,
#         "edge_index_dict": edge_index_dict,
#     }


# def debinary_hetero_data(data: dict):
#     x_dict = data["x_dict"].copy()
#     edge_index_dict = data["edge_index_dict"].copy()

#     for k, v in x_dict.items():
#         x_dict[k] = load_binary(v)
#     for k, v in edge_index_dict.items():
#         edge_index_dict[k] = load_binary(v)

#     return {"x_dict": x_dict, "edge_index_dict": edge_index_dict}
