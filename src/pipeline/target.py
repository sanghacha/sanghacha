"""Resolve an appearance request to an episode-local track, without simulator IDs."""
from dataclasses import replace

import numpy as np

from .types import SceneGraph, TaskRequest, check


class TargetResolver:
    def __init__(self):
        self.episode = None
        self.object_id = None

    def resolve(self, graph: SceneGraph, task: TaskRequest) -> TaskRequest:
        if task.target_spec is None:
            return task
        if graph.episode_id != self.episode:
            self.episode, self.object_id = graph.episode_id, None
        if self.object_id is not None:
            return replace(task, target_object_id=self.object_id)
        spec = task.target_spec
        ref = np.asarray(spec.reference_rgb) / sum(spec.reference_rgb)
        candidates = []
        for obj in graph.nodes:
            canonical = lambda label: 'box' if label in ('box', 'cube', 'block') else label
            if obj.color_rgb is None or canonical(obj.label) != canonical(spec.category):
                continue
            parent = obj.parent_id
            ancestors = set()
            while parent is not None:
                ancestors.add(parent)
                parent = graph.groups.get(parent)
            if spec.region and spec.region not in ancestors:
                continue
            if spec.size_m is not None and not np.allclose(obj.size, spec.size_m, atol=.008, rtol=0):
                continue
            rgb = np.asarray(obj.color_rgb)
            distance = float(np.linalg.norm(rgb / max(float(rgb.sum()), 1e-6) - ref))
            if distance <= spec.color_tolerance:
                candidates.append((distance, obj.object_id))
        candidates.sort()
        if not candidates:
            return task  # B may search; never manufacture a target pose/ID.
        check(len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= spec.ambiguity_margin,
              "ambiguous target: provide a more specific color, region or size")
        self.object_id = candidates[0][1]
        return replace(task, target_object_id=self.object_id)
