import importlib.util
from pathlib import Path
import unittest

from pipeline.mock_data import ROOT

spec=importlib.util.spec_from_file_location('benchmark_report',ROOT/'scripts/report_planner_benchmark.py')
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def row(case,method,success,removals=0):
    return dict(case_id=case,method=method,success=success,completed_removals=removals,
        simulation_seconds=10*removals,error=None if success else 'subgoal step budget exhausted',
        trace=[],observational_complete=success,group='visible_blocked',
        discovery_after_completed_removals=None,attempted_picks=removals+1,
        wall_seconds=1.,plan_seconds=.01,perception_seconds=.1)


class BenchmarkReportTests(unittest.TestCase):
    def test_discovery_counts_occluder_pick_even_before_buffer_place(self):
        observed=dict(discovered_at_sim_seconds=10.,trace=[
            dict(event='planned',primitive='PICK',target=False,timestamp=0.),
            dict(event='completed',primitive='PICK',target=False,timestamp=10.),
            dict(event='completed',primitive='PLACE',target=False,timestamp=20.)])
        self.assertEqual(report.discovery_picks(observed),1)
        observed['discovered_at_sim_seconds']=None
        self.assertIsNone(report.discovery_picks(observed))

    def test_failure_is_censored_in_efficiency_and_paired_difference_uses_same_scenes(self):
        left=[row('one','hsg',True,1),row('two','hsg',False),row('three','hsg',True,3)]
        right=[row('one','flat',True,2),row('two','flat',True,2),row('three','flat',False)]
        result=report.paired(left,right)
        self.assertEqual(result['left_only_success'],1)
        self.assertEqual(result['right_only_success'],1)
        self.assertEqual(result['both_success_n'],1)
        self.assertEqual(result['mean_removal_delta_both_success'],-1)
        summary=report.summarize(left+right)
        self.assertEqual(summary['by_method']['hsg']['all']['mean_removals_success'],2)
        self.assertEqual(summary['by_method']['hsg']['all']['target_never_correctly_bound_n'],3)

    def test_observation_proxy_does_not_override_physical_failure(self):
        failed=row('one','hsg',False)
        failed.update(observational_complete=True,error=None)
        self.assertEqual(report.failure(failed),'observational_completion_failed_physical_audit')
        result=report.summarize([failed])['by_method']['hsg']['all']
        self.assertEqual(result['successes'],0)
        self.assertEqual(result['false_observational_completion'],1)

    def test_wilson_interval_keeps_uncertainty_for_zero_and_perfect_success(self):
        self.assertGreater(report.wilson(0,13)[1],0)
        self.assertLess(report.wilson(13,13)[0],1)


if __name__=='__main__':
    unittest.main()
