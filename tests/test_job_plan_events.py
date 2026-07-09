# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structural tests for JobPlanStepEventSignal and JobPlanStepEventWait.

These tests do not require a Spyre device or the flex-1 runtime methods
(launchOperationEventSignal / launchOperationEventWait). They verify that:
  - the step classes are correctly registered in get_step_type()
  - write() produces output containing the expected event_id
"""

import pytest
from torch.testing._internal.common_utils import TestCase

import torch_spyre._C as _C


def _make_event_plan(signal_id: int, wait_id: int) -> _C.JobPlan:
    """Build a minimal JobPlan containing one EventSignal and one EventWait step.

    Uses the internal _C.make_event_plan helper exposed for testing.
    """
    return _C.make_event_plan(signal_id, wait_id)


class TestJobPlanEventSteps(TestCase):
    """Structural tests for event step classes — no device required."""

    def test_get_step_type_event_signal(self):
        """get_step_type() returns 'EventSignal' for a signal step."""
        plan = _C.make_event_plan(signal_id=1, wait_id=2)
        self.assertEqual(plan.get_step_type(0), "EventSignal")

    def test_get_step_type_event_wait(self):
        """get_step_type() returns 'EventWait' for a wait step."""
        plan = _C.make_event_plan(signal_id=1, wait_id=2)
        self.assertEqual(plan.get_step_type(1), "EventWait")

    def test_write_signal_contains_event_id(self):
        """str(plan) output for an EventSignal step contains the event_id."""
        plan = _C.make_event_plan(signal_id=7, wait_id=99)
        output = str(plan)
        self.assertIn("EventSignal", output)
        self.assertIn("7", output)

    def test_write_wait_contains_event_id(self):
        """str(plan) output for an EventWait step contains the event_id."""
        plan = _C.make_event_plan(signal_id=7, wait_id=99)
        output = str(plan)
        self.assertIn("EventWait", output)
        self.assertIn("99", output)

    def test_plan_has_two_steps(self):
        """A plan built with make_event_plan has exactly 2 steps."""
        plan = _C.make_event_plan(signal_id=0, wait_id=0)
        self.assertEqual(plan.num_steps(), 2)

    def test_same_event_id_signal_and_wait(self):
        """Signal and wait steps with the same ID are both present and typed."""
        plan = _C.make_event_plan(signal_id=5, wait_id=5)
        self.assertEqual(plan.get_step_type(0), "EventSignal")
        self.assertEqual(plan.get_step_type(1), "EventWait")
        output = str(plan)
        # Both steps reference event_id 5
        self.assertEqual(output.count("5"), 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
