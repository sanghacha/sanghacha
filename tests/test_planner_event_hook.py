from contextlib import redirect_stdout
import importlib.util
import io
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from pipeline.mock_data import ROOT
from planner.planner import plan_next_subgoal


class PlannerEventHookTests(unittest.TestCase):
    def test_completed_hook_is_delivered_without_gui(self):
        spec=importlib.util.spec_from_file_location('pipeline_cli_hook',ROOT/'scripts/run_pipeline.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        events=[]
        module=ModuleType('_planner_hook_test')
        module.plan_next_subgoal=plan_next_subgoal
        module.on_pipeline_event=lambda kind,**data:events.append((kind,data))
        with patch.dict(sys.modules,{'_planner_hook_test':module}), patch.object(sys,'argv',[
                'run_pipeline.py','--case','blocked','--planner-module','_planner_hook_test']),redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(),0)
        self.assertEqual(sum(kind=='completed' for kind,_ in events),4)
        self.assertEqual(events[-1][0],'finished')


if __name__=='__main__': unittest.main()
