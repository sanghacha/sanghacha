"""Describe changes between graph snapshots whose object IDs are supplied."""
import math
import numpy as np
from .validation import json_value


def graph_delta(previous, current, position_threshold_m=.005):
    """Compare snapshots without associating IDs or applying action commands."""
    if not math.isfinite(position_threshold_m) or position_threshold_m < 0:
        raise ValueError("position_threshold_m must be finite and nonnegative")
    if previous is not None:
        if previous.graph['episode_id'] != current.graph['episode_id']:
            raise ValueError("Cannot compare object IDs across different episodes")
        if previous.graph['frame_id'] != current.graph['frame_id']:
            raise ValueError("Cannot compare different coordinate frames")
        if current.graph['timestamp'] <= previous.graph['timestamp']:
            raise ValueError("The current graph must have a later timestamp")

    def relations(graph):
        return {(a, b, attrs['relation']) for a, b, attrs in graph.edges(data=True)}

    def membership(graph, key):
        return next((b for _, b, attrs in graph.out_edges(key, data=True) if attrs['relation'] == 'inside'), None)

    def relation_record(edge):
        a, b, relation = edge
        return dict(source=a, target=b, relation=relation)

    old = relations(previous) if previous is not None else set()
    new = relations(current)
    changes = []
    updated_relations = []
    if previous is not None:
        for key, node in current.nodes(data=True):
            if node.get('kind') != 'object_candidate' or key not in previous:
                continue
            before = previous.nodes[key]
            displacement = float(np.linalg.norm(np.array(node['obb']['center_m']) - before['obb']['center_m']))
            old_region, new_region = membership(previous, key), membership(current, key)
            if displacement > position_threshold_m or old_region != new_region or before['status'] != node['status']:
                changes.append(dict(instance_id=key, old_status=before['status'], new_status=node['status'],
                                    old_region=old_region, new_region=new_region,
                                    old_center_m=before['obb']['center_m'], new_center_m=node['obb']['center_m'],
                                    displacement_m=displacement,
                                    position_updated_from_current_observation=node['status'] == 'seen',
                                    tracking_evidence=node.get('tracking_evidence')))
        for edge in sorted(old & new):
            a, b, _ = edge
            old_attrs, new_attrs = previous.edges[a, b], current.edges[a, b]
            if old_attrs != new_attrs:
                updated_relations.append(dict(**relation_record(edge), old_attributes=dict(old_attrs), new_attributes=dict(new_attrs)))
    return json_value(dict(previous_timestamp=previous.graph['timestamp'] if previous is not None else None,
                timestamp=current.graph['timestamp'], episode_id=current.graph['episode_id'],
                position_threshold_m=position_threshold_m,
                added_nodes=sorted(set(current) - (set(previous) if previous is not None else set())),
                removed_nodes=sorted((set(previous) if previous is not None else set()) - set(current)),
                added_relations=[relation_record(edge) for edge in sorted(new - old)],
                removed_relations=[relation_record(edge) for edge in sorted(old - new)],
                updated_relations=updated_relations, object_changes=changes))
