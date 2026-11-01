"""Finite JSON input/output only; no simulator or point-cloud file access."""
import json
from pathlib import Path
import networkx as nx

from .validation import identifier, json_value, number

RELATIONS = {'inside', 'relative_pose_to', 'kinematic_child_of', 'fixed_in'}


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _read(path):
    return json_value(json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=_reject_constant))


def load_snapshot(path):
    """Read {objects: [...], metadata: {...}}; builder validates its contract."""
    value = _read(path)
    if not isinstance(value, dict) or not isinstance(value.get('objects'), list) or not isinstance(value.get('metadata'), dict):
        raise ValueError("Snapshot must contain objects (list) and metadata (object)")
    return value['objects'], value['metadata']


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_graph(graph, path):
    save_json(nx.node_link_data(graph, edges='edges'), path)


def load_graph(path):
    data = _read(path)
    if not isinstance(data, dict):
        raise ValueError("Graph JSON must be an object")
    if data.get('directed') is not True or data.get('multigraph') is not False:
        raise ValueError("Expected a directed, non-multigraph HSSG snapshot")
    meta = data['graph']
    if meta.get('frame_id') != 'robot_base' or meta.get('source') != 'rgbd':
        raise ValueError("Expected an RGB-D graph in robot_base")
    identifier(meta['episode_id'], 'episode_id')
    if number(meta['timestamp'], 'timestamp') < 0:
        raise ValueError("timestamp must be nonnegative")
    ids = [node['id'] for node in data['nodes']]
    if any(not isinstance(key, str) for key in ids) or len(ids) != len(set(ids)):
        raise ValueError("Graph node IDs must be unique strings")
    endpoints = set(ids)
    pairs = set()
    for edge in data['edges']:
        pair = (edge['source'], edge['target'])
        if pair in pairs or edge['source'] not in endpoints or edge['target'] not in endpoints:
            raise ValueError("Duplicate edge or unknown endpoint")
        pairs.add(pair)
        if edge['relation'] not in RELATIONS:
            raise ValueError("Unsupported relation")
    return nx.node_link_graph(data, edges='edges')
