"""Bind fixed-ACT comparison policies to the existing CLI planner interface."""
from .act_comparison import ComparisonPlanner, distance
from .planner import _on_top_of


def bind(condition):
    policy=ComparisonPlanner(condition)
    def event(kind,**payload):
        if kind=='completed': policy.completed(payload['subgoal'])
    def explain(graph,task,subgoal):
        mode=policy.last_decision.get('mode','search')
        if mode=='place':
            mode='target_place' if subgoal.object_id==task.target_object_id else 'buffer_place'
            reason='세 조건 공통 배치 규칙: 숨은 공간 점수를 사용하지 않습니다.'
        elif mode=='target_pick':
            reason='현재 관측에서 위가 빈 목표를 우선 집습니다.'
        elif mode=='unblock' and condition!='a_greedy':
            reason='관측된 적층 관계를 따라 목표 위 방해물을 제거합니다.'
        elif condition=='c_full':
            reason='추정 숨은 공간을 우선 탐색합니다. 단서가 없으면 로봇 기준점에 가까운 후보를 선택합니다.'
        else:
            reason=f'robot_base 원점에 가까운 후보부터 선택합니다. 선택 거리 {distance(graph.object(subgoal.object_id)):.3f} m'
        pairs=sorted(_on_top_of(graph)) if policy.engine is not None else []
        candidates=[]
        if condition=='c_full' and mode=='search':
            candidates=[dict(object_id=row['object_id'],hidden=row['hidden_count'],
                             shelf_hidden=row['region_hidden_count'])
                        for row in policy.last_decision.get('candidates',[])]
        return dict(mode=mode,reason=reason,support_pairs=pairs,
                    support_source='geometry' if policy.engine is not None else 'none',chain=[],candidates=candidates)
    return policy.plan_next_subgoal,event,explain
