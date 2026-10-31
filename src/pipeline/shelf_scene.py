"""세 개의 낮은 선반과 seed로 재현 가능한 안정적 박스 적재 배치."""

import colorsys

import mujoco
import numpy as np

from .types import check


def add_empty_tray(world, tray_id, center, size, floor_thickness, wall_thickness, wall_height,
                   frame_color, floor_color=None):
    """바닥과 네 개의 낮은 벽을 가진, 실제 충돌 가능한 빈 선반."""
    check(all(np.isfinite(value) and value > 0 for value in
              (*size, floor_thickness, wall_thickness, wall_height)), "invalid tray dimensions")
    check(min(size) > 2*wall_thickness, "tray walls leave no inner space")
    x, y = center
    sx, sy = size
    common = {"type": mujoco.mjtGeom.mjGEOM_BOX, "friction": [1., .01, .001]}
    world.add_geom(name=f"{tray_id}_floor", pos=[x, y, floor_thickness/2],
                   size=[sx/2, sy/2, floor_thickness/2],
                   rgba=floor_color if floor_color is not None else frame_color, **common)
    for sign in (-1, 1):
        world.add_geom(name=f"{tray_id}_wall_x_{sign}",
                       pos=[x + sign*(sx-wall_thickness)/2, y, floor_thickness+wall_height/2],
                       size=[wall_thickness/2, sy/2, wall_height/2], rgba=frame_color, **common)
        world.add_geom(name=f"{tray_id}_wall_y_{sign}",
                       pos=[x, y + sign*(sy-wall_thickness)/2, floor_thickness+wall_height/2],
                       size=[sx/2-wall_thickness, wall_thickness/2, wall_height/2], rgba=frame_color, **common)


class ShelfScene:
    def __init__(self, config):
        self.config = config
        self.shelves = config["shelves"]
        check(len(self.shelves) == 3, "expected three shelves")
        check(len({s["id"] for s in self.shelves}) == 3, "duplicate shelf IDs")
        count = config["objects_per_shelf"]
        check(type(count) is int and count > 0, "invalid objects_per_shelf")
        check(type(config["max_stack_height"]) is int and 1 <= config["max_stack_height"] <= 3,
              "max_stack_height must be 1..3")
        check(type(config["min_stacks"]) is int and 1 <= config["min_stacks"] <= 6, "invalid min_stacks")
        check(np.isfinite(config["density"]) and config["density"] > 0, "invalid density")
        check(0 <= config["yaw_range"] <= np.pi/4 and config["position_jitter"] >= 0, "invalid pose jitter")
        for size in config["object_sizes"]:
            check(len(size) == 3 and np.isfinite(size).all() and min(size) > 0, "invalid object size")
            check(np.isclose(size[0], size[1]), "starter shelf layout uses square XY footprints")
        check(bool(config["object_sizes"]), "object_sizes cannot be empty")
        half_extent = max(size[0] for size in config["object_sizes"]) / np.sqrt(2)
        check(2*(half_extent + config["position_jitter"]) < .05,
              "objects/jitter are too large for separated shelf slots")
        for shelf in self.shelves:
            check(shelf["id"] not in ("workspace", "bin", "tabletop"), "reserved shelf ID")
            check(len(shelf["center"]) == 2 and np.isfinite(shelf["center"]).all(), "invalid shelf center")
            check(len(shelf["size"]) == 2 and np.isfinite(shelf["size"]).all(), "invalid shelf size")
            available = np.array(shelf["size"])/2 - config["wall_thickness"]
            check(np.all(available > np.array([.05, .0275]) + half_extent + config["position_jitter"]),
                  "shelf is too small for the configured objects")
        check(count <= 6 * config["max_stack_height"], "too many objects for six shelf slots")
        self.objects = {}
        for shelf_index, shelf in enumerate(self.shelves):
            for index in range(count):
                # XY 크기를 내림차순으로 놓으면 작은 상자가 큰 상자 위에 쌓인다.
                sizes = config["object_sizes"]
                size = tuple(sizes[index % len(sizes)])
                key = f"{shelf['id']}_obj_{index+1:02d}"
                hue = (index * .61803398875 + shelf_index * .19) % 1
                self.objects[key] = {"size": size, "shelf_id": shelf["id"],
                                     "mass": config["density"] * float(np.prod(size)),
                                     "color": [*colorsys.hsv_to_rgb(hue, .82, .9), 1.]}
        check(config["target_object_id"] in self.objects, "unknown shelf target")
        # 모델 생성 시에도 겹치지 않는 초기 좌표를 사용한다. reset에서는 요청 seed로 다시 배치한다.
        for key, (position, quaternion) in self.sample_layout(np.random.default_rng(0)).items():
            self.objects[key]["position"] = position
            self.objects[key]["quaternion"] = quaternion

    def add_shelves(self, world):
        c = self.config
        for shelf in self.shelves:
            add_empty_tray(world, shelf["id"], shelf["center"], shelf["size"],
                           c["floor_thickness"], c["wall_thickness"], c["wall_height"], [.12, .14, .17, 1.])

    def sample_layout(self, rng):
        c = self.config
        result = {}
        for shelf in self.shelves:
            ids = [key for key, obj in self.objects.items() if obj["shelf_id"] == shelf["id"]]
            # 큰 면적의 물체부터 배치하되 동일 크기의 물체 순서는 seed에 따라 섞는다.
            rng.shuffle(ids)
            ids.sort(key=lambda key: self.objects[key]["size"][0] * self.objects[key]["size"][1], reverse=True)
            minimum = max(c["min_stacks"], int(np.ceil(len(ids)/c["max_stack_height"])))
            maximum = min(6, len(ids))
            check(minimum <= maximum, "invalid stack count")
            n_stacks = int(rng.integers(minimum, maximum+1))
            cells = [(x, y) for x in (-.05, 0., .05) for y in (-.0275, .0275)]
            rng.shuffle(cells)
            columns = []
            for index in range(n_stacks):
                jitter = rng.uniform(-c["position_jitter"], c["position_jitter"], 2)
                center = np.array(shelf["center"]) + cells[index] + jitter
                columns.append({"center": center, "yaw": rng.uniform(-c["yaw_range"], c["yaw_range"]),
                                "height": c["floor_thickness"], "ids": []})
            for index, key in enumerate(ids):
                if index < n_stacks:
                    column = columns[index]
                else:
                    candidates = [col for col in columns if len(col["ids"]) < c["max_stack_height"]]
                    target_column = next((col for col in candidates if c["target_object_id"] in col["ids"] and
                                          len(col["ids"]) == 1), None)
                    column = target_column if target_column is not None else candidates[int(rng.integers(len(candidates)))]
                obj = self.objects[key]
                half_z = obj["size"][2]/2
                position = [*column["center"], column["height"] + half_z + .0002]
                yaw = column["yaw"]
                quaternion = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
                result[key] = (position, quaternion)
                column["height"] += obj["size"][2] + .0002
                column["ids"].append(key)
        return result

    def region_of(self, position):
        x, y = position[:2]
        for shelf in self.shelves:
            if all(abs(value-center) <= size/2 for value, center, size in zip((x, y), shelf["center"], shelf["size"])):
                return shelf["id"]
        return "tabletop"
