"""Evidence must distinguish admission, execution and observation failures."""
import copy
import unittest
from unittest.mock import patch

import torch
from stepcontract import StepContract, UnsupportedOperation


def setup(optimizer=torch.optim.SGD):
    m = torch.nn.Linear(2, 1, bias=False)
    o = optimizer(m.parameters(), lr=.1)
    return m, o, StepContract(m, o, required_parameters=['weight'])


def cycle(m, a):
    a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum()); a.step()


class EventEvidenceTests(unittest.TestCase):
    def test_snapshot_failure_is_sticky_error_without_step(self):
        m, o, a = setup()
        cycle(m, a)
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum())
        before = m.weight.detach().clone()
        with patch.object(torch.Tensor, 'clone', side_effect=RuntimeError('injected snapshot failure')):
            with self.assertRaisesRegex(RuntimeError, 'snapshot failure'):
                a.step()
        r = a.report()
        self.assertEqual(r['status'], 'ERROR')
        self.assertEqual(r['summary']['completed_steps'], 1)
        self.assertTrue(torch.equal(before, m.weight))
        failed = r['events'][-1]
        self.assertEqual(failed['outcome'], 'failed')
        self.assertFalse(failed['operation_started'])
        a.step()
        r = a.report()
        self.assertEqual(r['status'], 'ERROR')
        self.assertEqual(r['summary']['completed_steps'], 2)
        self.assertEqual(r['events'][-1]['outcome'], 'completed')
        error = next(f for f in r['findings'] if f['rule'] == 'execution_error')
        self.assertEqual(error['event_index'], failed['index'])

    def test_early_zero_links_actual_event(self):
        m, _, a = setup()
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum()); a.zero_grad(); a.step()
        r = a.report()
        f = next(f for f in r['findings'] if f['rule'] == 'early_zero_grad')
        self.assertEqual(f['event_index'], 2)
        self.assertEqual(r['events'][f['event_index']]['kind'], 'zero_grad')
        self.assertEqual(f['origin'], 'event')

    def test_constructor_and_report_observations_have_no_fake_event(self):
        m = torch.nn.Linear(2, 1, bias=False)
        o = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=.1)
        a = StepContract(m, o, required_parameters=['weight'])
        f = next(f for f in a.report()['findings'] if f['rule'] == 'missing_optimizer_parameter')
        self.assertIsNone(f['event_index'])
        self.assertEqual(f['origin'], 'construction')
        m, _, a = setup()
        cycle(m, a)
        m.weight.requires_grad_(False)
        f = next(f for f in a.report()['findings'] if f['rule'] == 'required_parameter_frozen')
        self.assertIsNone(f['event_index'])
        self.assertEqual(f['origin'], 'report')

    def test_dynamic_modes_are_refused_before_execution(self):
        for flag in ('fused', 'capturable', 'differentiable'):
            for operation in ('zero_grad', 'backward', 'step', 'report'):
                with self.subTest(flag=flag, operation=operation):
                    m, o, a = setup(torch.optim.AdamW)
                    cycle(m, a)
                    before_events = copy.deepcopy(a.report()['events'])
                    before = m.weight.detach().clone()
                    o.param_groups[0][flag] = True
                    if operation == 'report':
                        self.assertEqual(a.report()['status'], 'UNSUPPORTED')
                    else:
                        with self.assertRaises(UnsupportedOperation):
                            if operation == 'backward':
                                a.backward(m(torch.ones(1, 2)).sum())
                            else:
                                getattr(a, operation)()
                    self.assertEqual(a.report()['status'], 'UNSUPPORTED')
                    self.assertEqual(a.report()['events'], before_events)
                    self.assertEqual(a.report()['summary']['completed_steps'], 1)
                    self.assertTrue(torch.equal(m.weight, before))

    def test_legal_learning_rate_change_not_refused(self):
        m, o, a = setup(torch.optim.AdamW)
        cycle(m, a)
        o.param_groups[0]['lr'] = .03
        cycle(m, a)
        self.assertEqual(a.report()['status'], 'PASS')

    def test_rejected_invalid_argument_does_not_mutate_history(self):
        m, _, a = setup()
        cycle(m, a)
        prior = a.report()
        with self.assertRaises(ValueError):
            a.zero_grad(set_to_none='invalid')
        self.assertEqual(a.report(), prior)

    def test_post_step_observation_error_reports_execution_happened(self):
        m, o, a = setup()
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum())
        before = m.weight.detach().clone()
        with patch('stepcontract.audit.torch.equal', side_effect=RuntimeError('injected observation failure')):
            with self.assertRaisesRegex(RuntimeError, 'observation failure'):
                a.step()
        r = a.report()
        self.assertEqual(r['status'], 'ERROR')
        self.assertEqual(r['summary']['completed_steps'], 0)
        self.assertTrue(r['events'][-1]['operation_returned'])
        self.assertFalse(torch.equal(m.weight, before))


if __name__ == '__main__':
    unittest.main()
