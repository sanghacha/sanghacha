"""Public B adapter: --planner-module planner.hsg_incremental."""
from .indexed import IndexedPlanner
from .planner import explain_subgoal as _explain

_planner = IndexedPlanner('hsg_incremental')


def plan_next_subgoal(graph,task,robot_state):
    return _planner.plan_next_subgoal(graph,task,robot_state)


def explain_subgoal(graph,task,subgoal):
    result = _explain(graph,task,subgoal)
    stats = _planner.stats
    result['reason'] += (f'\nHSG: 변경 {stats.changed_objects}개 · '
                         f'갱신 영역 {len(stats.dirty_regions)}개')
    return result
