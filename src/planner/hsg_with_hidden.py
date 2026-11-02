"""Condition C: HSG with inferred hidden-space cues + fixed ACT."""
from .comparison_adapter import bind
plan_next_subgoal,on_pipeline_event,explain_subgoal=bind('c_full')
