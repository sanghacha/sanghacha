- src
    - hssg_generator
    - planner
    - action_generator
    - pipeline
- configs
    - envs
    - io
- scripts


입출력
```
# 모듈 A
def generate_scene_graph(
    observation: Observation
) -> SceneGraph:
    ...

# 모듈 B
def plan_next_subgoal(
    graph: SceneGraph,
    task,
    robot_state
) -> Subgoal:
    ...

# 모듈 C
def predict_action_chunk(
    observation: Observation,
    subgoal: Subgoal
) -> ActionChunk:
    ...
```