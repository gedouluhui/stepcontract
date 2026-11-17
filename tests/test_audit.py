import copy
import importlib.util
import unittest

import torch


def model():
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1., 2.]]))
    return layer


class AuditTests(unittest.TestCase):
    def make(self, m=None, opt=None, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('stepcontract'), 'package must exist')
        import stepcontract
        self.assertTrue(callable(getattr(stepcontract, 'StepContract', None)), 'StepContract API must exist')
        m = m if m is not None else model()
        opt = opt if opt is not None else torch.optim.SGD(m.parameters(), lr=0.1)
        return m, opt, stepcontract.StepContract(m, opt, required_parameters=kwargs.pop('required_parameters', ['weight']), **kwargs)

    def rules(self, audit):
        return {f['rule'] for f in audit.report()['findings']}

    def test_healthy_and_exact_control(self):
        for kind in ('SGD', 'AdamW'):
            with self.subTest(optimizer=kind):
                m = model()
                control = copy.deepcopy(m)
                cls = getattr(torch.optim, kind)
                o, co = cls(m.parameters(), lr=0.1), cls(control.parameters(), lr=0.1)
                _, _, a = self.make(m, o)
                a.zero_grad()
                a.backward(m(torch.ones(1, 2)).sum())
                a.step()
                co.zero_grad()
                control(torch.ones(1, 2)).sum().backward()
                co.step()
                self.assertEqual(a.report()['status'], 'PASS')
                self.assertTrue(torch.equal(m.weight, control.weight))
                self.assertTrue(torch.equal(m.weight.grad, control.weight.grad))

    def test_missing_optimizer_parameter(self):
        m = torch.nn.Sequential(model(), torch.nn.Linear(1, 1, bias=False))
        o = torch.optim.SGD(m[0].parameters(), lr=0.1)
        _, _, a = self.make(m, o, required_parameters=['0.weight', '1.weight'])
        self.assertIn('missing_optimizer_parameter', self.rules(a))
        self.assertEqual(a.report()['status'], 'FAIL')

    def test_required_frozen(self):
        m = model()
        m.weight.requires_grad_(False)
        _, _, a = self.make(m)
        self.assertIn('required_parameter_frozen', self.rules(a))

    def test_legal_freeze(self):
        m = torch.nn.Sequential(model(), torch.nn.Linear(1, 1, bias=False))
        m[1].weight.requires_grad_(False)
        _, _, a = self.make(m, required_parameters=['0.weight'])
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum()); a.step()
        self.assertEqual(a.report()['status'], 'PASS')

    def test_early_zero_records_order(self):
        m, _, a = self.make()
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum()); a.zero_grad(); a.step()
        self.assertEqual(a.report()['status'], 'FAIL')
        self.assertIn('early_zero_grad', self.rules(a))
        self.assertEqual([e['kind'] for e in a.report()['events']], ['zero_grad', 'backward', 'zero_grad', 'step'])

    def test_legal_accumulation(self):
        m, _, a = self.make(accumulation_steps=2)
        a.zero_grad()
        a.backward(m(torch.ones(1, 2)).sum())
        a.backward(m(torch.ones(1, 2)).sum())
        a.step()
        self.assertEqual(a.report()['status'], 'PASS')

    def test_wrong_accumulation_count(self):
        m, _, a = self.make(accumulation_steps=2)
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum()); a.step()
        self.assertIn('accumulation_count', self.rules(a))

    def test_unfinished_is_incomplete(self):
        m, _, a = self.make()
        self.assertEqual(a.report()['status'], 'INCOMPLETE')
        a.backward(m(torch.ones(1, 2)).sum())
        self.assertEqual(a.report()['status'], 'INCOMPLETE')

    def test_nonfinite_gradient(self):
        for value in (float('nan'), float('inf')):
            m, _, a = self.make()
            a.zero_grad(); a.backward(m(torch.tensor([[value, 1.]])).sum()); a.step()
            self.assertEqual(a.report()['status'], 'FAIL')
            self.assertIn('nonfinite_gradient', self.rules(a))

    def test_zero_learning_rate_is_not_failure(self):
        m = model()
        _, _, a = self.make(m, torch.optim.SGD(m.parameters(), lr=0))
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum()); a.step()
        self.assertEqual(a.report()['status'], 'PASS')
        self.assertFalse(a.report()['parameters'][0]['changed_last_step'])

    def test_zero_gradient_is_not_failure(self):
        m, _, a = self.make()
        a.zero_grad(); a.backward(m(torch.zeros(1, 2)).sum()); a.step()
        self.assertEqual(a.report()['status'], 'PASS')

    def test_residue_after_step_without_clearing(self):
        m, _, a = self.make()
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum()); a.step()
        a.backward(m(torch.ones(1, 2)).sum()); a.step()
        self.assertIn('residual_gradient', self.rules(a))

    def test_external_clear_has_evidence_not_guessed_cause(self):
        m, o, a = self.make()
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum()); o.zero_grad(); a.step()
        self.assertIn('gradient_disappeared', self.rules(a))

    def test_unknown_name_bad_limits_and_budget(self):
        import stepcontract
        self.assertTrue(hasattr(stepcontract, 'StepContract'), 'StepContract API must exist')
        for kwargs in ({'required_parameters': ['absent']}, {'required_parameters': []},
                       {'accumulation_steps': True}, {'max_snapshot_bytes': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)

    def test_parameter_replacement_detected(self):
        m, _, a = self.make()
        m.weight = torch.nn.Parameter(torch.ones(1, 2))
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum()); a.step()
        self.assertIn('parameter_identity_changed', self.rules(a))

    def test_rng_not_consumed(self):
        m = model()
        o = torch.optim.SGD(m.parameters(), lr=.1)
        rng = torch.get_rng_state().clone()
        _, _, a = self.make(m, o)
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum()); a.step(); a.report()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_report_contains_no_tensors(self):
        import json
        m, _, a = self.make()
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum()); a.step()
        data = json.dumps(a.report(), allow_nan=False)
        self.assertNotIn('tensor(', data)
        self.assertNotIn('raw_gradient', data)

    def test_clear_between_accumulated_backwards_is_observed(self):
        m, o, a = self.make(accumulation_steps=2)
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum())
        o.zero_grad()
        a.backward(m(torch.ones(1, 2)).sum()); a.step()
        self.assertEqual(a.report()['status'], 'FAIL')
        self.assertIn('gradient_disappeared', self.rules(a))

    def test_storage_alias_added_after_binding_is_rejected(self):
        from stepcontract import UnsupportedOperation
        m = torch.nn.Module()
        m.p = torch.nn.Parameter(torch.tensor([1.]))
        m.q = torch.nn.Parameter(torch.tensor([2.]))
        _, _, a = self.make(m, required_parameters=['p', 'q'])
        m.q.data = m.p.data
        with self.assertRaises(UnsupportedOperation):
            a.zero_grad()
        self.assertEqual(a.report()['status'], 'UNSUPPORTED')

    def test_partially_overlapping_buffers_rejected(self):
        from stepcontract import UnsupportedOperation
        data = bytearray(12)
        m = torch.nn.Module()
        m.p = torch.nn.Parameter(torch.frombuffer(data, dtype=torch.float32, count=2, offset=0))
        m.q = torch.nn.Parameter(torch.frombuffer(data, dtype=torch.float32, count=2, offset=4))
        with self.assertRaises(UnsupportedOperation):
            self.make(m, required_parameters=['p', 'q'])

    def test_bad_backward_after_success_does_not_reuse_pass(self):
        m, _, a = self.make()
        a.zero_grad(); a.backward(m(torch.ones(1, 2)).sum()); a.step()
        self.assertEqual(a.report()['status'], 'PASS')
        a.zero_grad()
        with self.assertRaises(AttributeError):
            a.backward(1.0)
        self.assertEqual(a.report()['status'], 'ERROR')

    def test_backward_exception_remains_error(self):
        _, _, a = self.make()
        with self.assertRaises(RuntimeError):
            a.backward(torch.tensor(1.0))
        self.assertEqual(a.report()['status'], 'ERROR')


if __name__ == '__main__':
    unittest.main()
