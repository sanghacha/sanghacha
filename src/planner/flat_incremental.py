"""Same policy, flat storage and exact incremental relation maintenance."""
from .indexed import IndexedPlanner
from .planner import explain_subgoal
_planner = IndexedPlanner('flat_incremental')
plan_next_subgoal = _planner.plan_next_subgoal
