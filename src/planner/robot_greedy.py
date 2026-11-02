"""Condition A: nearest-to-robot-base Greedy + fixed downstream ACT."""
from .comparison_adapter import bind
plan_next_subgoal,on_pipeline_event,explain_subgoal=bind('a_greedy')
