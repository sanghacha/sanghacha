"""Same policy, regional summaries rebuilt on each graph."""
from .indexed import IndexedPlanner
from .planner import explain_subgoal
_planner = IndexedPlanner('hsg_full')
plan_next_subgoal = _planner.plan_next_subgoal
