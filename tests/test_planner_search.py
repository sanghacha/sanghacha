"""Guard against paired-scene drift and misleading diagnostic statistics."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_planner_search import check_initial_graph, blockers, nearest_physical
from report_planner_search import holm, paired_delta, summarize, digest, CONDITIONS
from audit_planner_search import ancestors
from benchmark_planner_occlusion import permute_columns
from pipeline.types import Edge, Pose, SceneGraph, SceneObject


class PlannerSearchTests(unittest.TestCase):
    def graph(self):
        obj=SceneObject('instance_0001',Pose((.24,0,.04)),(.028,)*3,
                        parent_id='shelf',color_rgb=(.8,.2,.1))
        return SceneGraph((obj,),(),{'shelf':None},source='rgbd')

    def test_initial_graph_tolerates_only_roundoff_not_a_different_scene(self):
        graph=self.graph();expected=json.loads(json.dumps(asdict(graph)))
        rounded=replace(graph,nodes=(replace(graph.nodes[0],color_rgb=(.800004,.2,.1)),))
        self.assertAlmostEqual(check_initial_graph(rounded,expected),.000004)
        moved=replace(graph,nodes=(replace(graph.nodes[0],pose=Pose((.241,0,.04))),))
        with self.assertRaises(AssertionError): check_initial_graph(moved,expected)
        wrong_id=replace(graph,nodes=(replace(graph.nodes[0],object_id='instance_0002'),))
        with self.assertRaises(AssertionError): check_initial_graph(wrong_id,expected)

    def test_executor_does_not_map_a_bad_pose_or_reuse_removed_object(self):
        class Sim:
            object_bodies={'one':0,'two':1}
            class data:
                xpos=np.array([[.2,0,.02],[.2,0,.06]])
        sim=Sim()
        self.assertEqual(nearest_physical(sim,(.2,0,.02),set()),'one')
        self.assertIsNone(nearest_physical(sim,(.2,0,.02),{'one'}))
        self.assertIsNone(nearest_physical(sim,(.5,0,.02),set()))

    def test_support_gate_respects_remaining_physical_blockers(self):
        graph=replace(self.graph(),nodes=self.graph().nodes+(
            SceneObject('top',Pose((.24,0,.068)),(.028,)*3,parent_id='shelf'),),
            edges=(Edge('top','instance_0001','on_top_of'),))
        self.assertEqual(blockers(graph,'instance_0001',set()),['top'])
        self.assertEqual(blockers(graph,'instance_0001',{'top'}),[])

    def test_pairing_is_by_case_not_input_order_and_zero_effect_stays_zero(self):
        result=paired_delta({'b':2,'a':1},{'a':1,'b':2})
        self.assertEqual(result['mean_saved'],0)
        self.assertEqual(result['mean_saved_bootstrap_95'],[0.,0.])
        self.assertEqual(result['exact_two_sided_sign_p'],1.)
        self.assertEqual(result['ties'],2)
        with self.assertRaises(AssertionError): paired_delta({'a':1},{'b':1})

    def test_sign_test_and_direction_do_not_turn_losses_into_wins(self):
        a={str(i):10 for i in range(6)};b={str(i):1 for i in range(6)}
        result=paired_delta(a,b)
        self.assertEqual(result['right_wins'],6)
        self.assertEqual(result['mean_saved'],9)
        self.assertAlmostEqual(result['exact_two_sided_sign_p'],.03125)
        reverse=paired_delta(b,a)
        self.assertEqual(reverse['mean_saved'],-9)
        self.assertEqual(reverse['left_wins'],6)

    def test_holm_controls_the_two_primary_comparisons(self):
        self.assertEqual(holm({'one':.02,'two':.03}),{'one':.04,'two':.04})
        self.assertEqual(holm({'one':.7,'two':.9}),{'one':1.,'two':1.})

    def test_oracle_lower_bound_counts_unique_required_blockers(self):
        self.assertEqual(ancestors([('a','target'),('b','a'),('b','target')],'target'),{'a','b'})
        self.assertEqual(ancestors([('a','elsewhere')],'target'),set())
        with self.assertRaises(AssertionError): ancestors([('a','target'),('target','a')],'target')

    def test_stress_sampler_preserves_columns_counts_and_vertical_support(self):
        definitions={'a':{'size':(.02,.02,.02)},'b':{'size':(.04,.04,.04)},'c':{'size':(.03,.03,.03)}}
        layout={'a':([.2,0,.016],[1,0,0,0]),'b':([.2,0,.046],[1,0,0,0]),
                'c':([.3,0,.021],[1,0,0,0])}
        result=permute_columns(layout,definitions,.006,np.random.default_rng(13))
        self.assertEqual(set(result),set(layout))
        for key in result: self.assertEqual(result[key][0][:2],layout[key][0][:2])
        stacked=sorted(('a','b'),key=lambda k:result[k][0][2])
        lower,upper=stacked
        gap=result[upper][0][2]-definitions[upper]['size'][2]/2-(result[lower][0][2]+definitions[lower]['size'][2]/2)
        self.assertAlmostEqual(gap,.0002)

    def test_report_requires_every_pair_and_includes_failures_in_cost(self):
        manifest={'cases':[{'case_id':'one'},{'case_id':'two'}], 'protocol':{'source_sha256':{}}}
        rows=[]
        for case,group in [('one','visible_blocked'),('two','unlocalized_unknown_region')]:
            for method in CONDITIONS:
                failed=case=='two' and method=='c_full'
                rows.append(dict(case_id=case,condition=method,group=group,success=not failed,
                    identity={'manifest_sha256':digest(manifest),'source_sha256':{}},
                    capped_removals=27 if failed else 1,non_target_removals=1,
                    discovery_removals=None if failed else 1,error='failure' if failed else None))
        result=summarize(rows,manifest)
        self.assertEqual(result['summary']['all']['c_full']['mean_capped_removals'],14.)
        self.assertEqual(result['summary']['all']['c_full']['successes'],1)
        with self.assertRaises(AssertionError): summarize(rows[:-1],manifest)
        with self.assertRaises(AssertionError): summarize(rows+[rows[0]],manifest)


if __name__=='__main__': unittest.main()
