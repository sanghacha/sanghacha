"""모듈 A 시작점: mock 기하 또는 실제 MuJoCo 접촉으로 GT 적층 그래프를 만든다."""

from dataclasses import replace

from pipeline.types import Edge, Observation, SceneGraph, check


def generate_scene_graph(observation: Observation) -> SceneGraph:
    check(observation.source in ("mock_gt", "mujoco_gt") and observation.ground_truth is not None,
          "starter generator requires explicit simulator ground truth")
    objects = observation.ground_truth
    regions = {obj.object_id: obj.parent_id or "bin" for obj in objects}
    edges = []
    components = [{obj.object_id} for obj in objects]
    for upper in objects:
        for lower in objects:
            if upper.object_id == lower.object_id or observation.held_object_id in (upper.object_id, lower.object_id):
                continue
            if regions[upper.object_id] != regions[lower.object_id]:
                continue
            a, b = upper.pose.position, lower.pose.position
            surface_gap = a[2] - upper.size[2]/2 - (b[2] + lower.size[2]/2)
            overlap = all(abs(a[i] - b[i]) < (upper.size[i] + lower.size[i])/2 for i in (0, 1))
            supported = overlap and abs(surface_gap) < 1e-4 and a[2] > b[2]
            if observation.source == "mujoco_gt":
                # 접촉, 수직 법선, 양의 접촉력, 위/아래 위치를 함께 확인한다.
                supported = a[2] > b[2] and abs(surface_gap) < .003 and any(
                    c.normal_force > 1e-5 and (
                        (c.entity_a == lower.object_id and c.entity_b == upper.object_id and c.normal_a_to_b[2] > .7) or
                        (c.entity_b == lower.object_id and c.entity_a == upper.object_id and c.normal_a_to_b[2] < -.7)
                    ) for c in observation.contacts)
            if supported:
                edges.extend((Edge(upper.object_id, lower.object_id, "on_top_of"),
                              Edge(lower.object_id, upper.object_id, "supports")))
                first = next(c for c in components if upper.object_id in c)
                second = next(c for c in components if lower.object_id in c)
                if first is not second:
                    first.update(second)
                    components.remove(second)
    groups = {"workspace": None, **{region: "workspace" for region in regions.values()}}
    parents = {}
    for component in components:
        group_id = "stack:" + ":".join(sorted(component))
        groups[group_id] = regions[next(iter(component))]
        parents.update({key: group_id for key in component})
    nodes = tuple(replace(obj, parent_id=parents[obj.object_id]) for obj in objects)
    return SceneGraph(nodes, tuple(edges), groups, observation.held_object_id,
                      observation.timestamp, observation.episode_id, observation.frame_id, observation.source)
