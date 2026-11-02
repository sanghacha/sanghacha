"""Same policy, flat storage and full rebuild on each graph."""
from .indexed import IndexedPlanner
from .planner import explain_subgoal
_planner = IndexedPlanner('flat_full')
plan_next_subgoal = _planner.plan_next_subgoal
