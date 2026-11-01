"""Auditable changes between RGB-D graph snapshots, with no GT inputs."""
import numpy as np


def graph_delta(previous,current):
    def relations(graph):
        return {(a,b,e['relation']) for a,b,e in graph.edges(data=True)}
    old=relations(previous) if previous is not None else set()
    new=relations(current)
    changes=[]
    if previous is not None:
        for key,node in current.nodes(data=True):
            if node.get('kind')!='object_candidate' or key not in previous:continue
            before=previous.nodes[key]
            delta=np.array(node['obb']['center_m'])-before['obb']['center_m']
            old_region=next((b for a,b,e in previous.out_edges(key,data=True) if e['relation']=='inside'),None)
            new_region=next((b for a,b,e in current.out_edges(key,data=True) if e['relation']=='inside'),None)
            if np.linalg.norm(delta)>.005 or old_region!=new_region or before['status']!=node['status']:
                changes.append(dict(instance_id=key,old_status=before['status'],new_status=node['status'],
                    old_region=old_region,new_region=new_region,old_center_m=before['obb']['center_m'],
                    new_center_m=node['obb']['center_m'],displacement_m=float(np.linalg.norm(delta)),
                    position_updated_from_current_observation=node['status']=='seen',
                    tracking_evidence=node.get('tracking_evidence')))
    return dict(previous_timestamp=previous.graph['timestamp'] if previous is not None else None,
                timestamp=current.graph['timestamp'],
                added_nodes=sorted(set(current)-(set(previous) if previous is not None else set())),
                added_relations=[dict(source=a,target=b,relation=r) for a,b,r in sorted(new-old)],
                removed_relations=[dict(source=a,target=b,relation=r) for a,b,r in sorted(old-new)],
                object_changes=changes)
