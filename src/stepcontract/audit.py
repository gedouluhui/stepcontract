"""Explicit, opt-in operation wrappers; no global hooks or hidden training."""
import copy
import torch

from .contract import UnsupportedOperation, Window, positive_int
from .snapshot import gradient_kind, named_parameters, unique_parameters, validate_parameter


class StepContract:
    def __init__(self, model, optimizer, *, required_parameters, accumulation_steps=1,
                 max_snapshot_bytes=67108864):
        self.model, self.optimizer = model, optimizer
        self.window = Window(positive_int(accumulation_steps, 'accumulation_steps'))
        self.budget = positive_int(max_snapshot_bytes, 'max_snapshot_bytes')
        if type(optimizer) not in (torch.optim.SGD, torch.optim.AdamW):
            raise UnsupportedOperation('only unmodified torch.optim.SGD and AdamW are supported')
        if any(g.get(k, False) for g in optimizer.param_groups for k in ('differentiable', 'capturable', 'fused')):
            raise UnsupportedOperation('differentiable, capturable and fused optimizer modes are unsupported')
        self.named = named_parameters(model)
        if (not isinstance(required_parameters, (list, tuple)) or not required_parameters
                or any(not isinstance(n, str) for n in required_parameters)
                or len(set(required_parameters)) != len(required_parameters)
                or not set(required_parameters) <= set(self.named)):
            raise ValueError('required_parameters must name an explicit nonempty unique subset of model parameters')
        self.required = list(required_parameters)
        self.unique = unique_parameters(self.named)
        self.snapshot_bytes = sum(r['parameter'].numel() * r['parameter'].element_size() for r in self.unique.values())
        if self.snapshot_bytes > self.budget:
            raise ValueError('exact parameter snapshot exceeds max_snapshot_bytes')
        self.events, self.findings = [], []
        self.failed_execution = False
        self.unsupported = False
        self.changed = {}
        self.after_backward = {}
        self.initial_ownership = self._ownership()
        self._check_identity()

    def _ownership(self):
        counts = {}
        for group in self.optimizer.param_groups:
            for parameter in group['params']:
                counts[id(parameter)] = counts.get(id(parameter), 0) + 1
        return counts

    def _finding(self, rule, parameter, detail):
        item = {'rule': rule, 'parameter': parameter, 'detail': detail, 'event_index': len(self.events)}
        if not any(f['rule'] == rule and f['parameter'] == parameter for f in self.findings):
            self.findings.append(item)

    def _check_identity(self):
        current = named_parameters(self.model)
        if set(current) != set(self.named) or any(current.get(n) is not p for n, p in self.named.items()):
            self._finding('parameter_identity_changed', '', 'Model parameter names or identities changed after the contract was bound.')
        for inventory in (current, self.named):
            try:
                unique_parameters(inventory)
            except UnsupportedOperation as exc:
                self.unsupported = True
                self._finding('unsupported_observation', '', str(exc))
        ownership = self._ownership()
        if ownership != self.initial_ownership:
            self._finding('optimizer_membership_changed', '', 'Optimizer membership changed after the contract was bound.')
        for name in self.required:
            p = self.named[name]
            if ownership.get(id(p), 0) == 0:
                self._finding('missing_optimizer_parameter', name, 'A parameter declared required is absent from the optimizer.')
            if not p.requires_grad:
                self._finding('required_parameter_frozen', name, 'A parameter declared trainable has requires_grad=False.')
        for identity, record in self.unique.items():
            p = record['parameter']
            if ownership.get(identity, 0) > 1:
                self._finding('duplicate_optimizer_parameter', record['names'][0], 'This Parameter has multiple optimizer ownership entries.')

    def _event(self, kind, **extra):
        if len(self.events) >= 10000:
            raise ValueError('visible event limit reached; use a new audit session')
        self._check_identity()
        if self.unsupported:
            raise UnsupportedOperation('model moved outside the supported observation scope; inspect report()')
        self.events.append({'index': len(self.events), 'kind': kind, **extra})

    def _gradients(self):
        result = {}
        for identity, record in self.unique.items():
            try:
                kind = gradient_kind(record['parameter'])
            except UnsupportedOperation as exc:
                self.unsupported = True
                self._finding('unsupported_observation', record['names'][0], str(exc))
                kind = 'unsupported'
            result[identity] = kind
            if kind == 'nonfinite':
                self._finding('nonfinite_gradient', record['names'][0], 'Observed gradient contains NaN or infinity.')
        return result

    def _execute(self, operation):
        try:
            if self.unsupported:
                raise UnsupportedOperation('operation refused outside supported observation scope')
            return operation()
        except UnsupportedOperation:
            self.unsupported = True
            raise
        except Exception as exc:
            self.failed_execution = True
            self._finding('execution_error', '', type(exc).__name__)
            raise

    def zero_grad(self, set_to_none=True):
        if type(set_to_none) is not bool:
            raise ValueError('set_to_none must be boolean')
        self._event('zero_grad', set_to_none=set_to_none)
        if self.window.clear():
            self._finding('early_zero_grad', '', 'zero_grad was called after backward and before optimizer step; pending gradients were discarded.')
        return self._execute(lambda: self.optimizer.zero_grad(set_to_none=set_to_none))

    def backward(self, loss):
        self._event('backward')
        before = self._gradients()
        if self.window.backward_count:
            for identity, kind in before.items():
                if self.after_backward.get(identity) == 'nonzero' and kind in ('none', 'zero'):
                    self._finding('gradient_disappeared', self.unique[identity]['names'][0],
                                  'A nonzero gradient became absent/zero between accumulated backward calls; the intervening cause is unobserved.')
        if self.window.backward_count == 0 and not self.window.cleared:
            for identity, kind in before.items():
                if kind in ('nonzero', 'nonfinite'):
                    self._finding('residual_gradient', self.unique[identity]['names'][0],
                                  'Nonzero gradient was present at the start of a new declared accumulation window.')
        self._execute(lambda: loss.backward())
        self.window.backward()
        self.after_backward = self._gradients()

    def step(self):
        self._event('step', backward_count=self.window.backward_count)
        if self.window.backward_count != self.window.accumulation_steps:
            self._finding('accumulation_count', '', 'Expected %d backward calls, observed %d.' %
                          (self.window.accumulation_steps, self.window.backward_count))
        now = self._gradients()
        for identity, kind in now.items():
            previous = self.after_backward.get(identity)
            if previous == 'nonzero' and kind in ('none', 'zero') and self.window.backward_count:
                self._finding('gradient_disappeared', self.unique[identity]['names'][0],
                              'A nonzero gradient after backward became absent/zero before step; the intervening cause is unobserved.')
        # Recheck the budget because tensor storage sizes can change in place.
        needed = sum(r['parameter'].numel() * r['parameter'].element_size() for r in self.unique.values())
        if needed > self.budget:
            self.failed_execution = True
            raise ValueError('parameter snapshot exceeds budget')
        before = {identity: r['parameter'].detach().clone() for identity, r in self.unique.items()}
        result = self._execute(self.optimizer.step)
        for identity, record in self.unique.items():
            p = record['parameter'].detach()
            self.changed[identity] = not torch.equal(before[identity], p)
            if not bool(torch.isfinite(p).all()):
                self._finding('nonfinite_parameter', record['names'][0], 'Parameter contains NaN or infinity after step.')
        self.window.step()
        return result

    def report(self):
        self._check_identity()
        gradients = self._gradients()
        status = ('ERROR' if self.failed_execution else 'UNSUPPORTED' if self.unsupported else
                  'FAIL' if self.findings else 'INCOMPLETE' if self.window.backward_count or not self.window.steps else 'PASS')
        ownership = self._ownership()
        parameters = [{'names': r['names'], 'shape': list(r['parameter'].shape),
                       'requires_grad': bool(r['parameter'].requires_grad),
                       'optimizer_memberships': ownership.get(identity, 0),
                       'gradient': gradients[identity], 'changed_last_step': self.changed.get(identity)}
                      for identity, r in self.unique.items()]
        return copy.deepcopy({'schema_version': 1, 'tool': 'stepcontract', 'version': '0.1.0', 'status': status,
                              'intent': {'required_parameters': self.required, 'accumulation_steps': self.window.accumulation_steps},
                              'events': self.events, 'parameters': parameters, 'findings': self.findings,
                              'summary': {'completed_steps': self.window.steps, 'unique_parameters': len(self.unique),
                                          'snapshot_bytes': self.snapshot_bytes},
                              'scope': 'Only explicit wrapper calls and bound CPU FP32 model parameters are observed. No AMP, distributed or compiled-graph certification; no assertion of global training correctness.'})
