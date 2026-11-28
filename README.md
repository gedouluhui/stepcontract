# StepContract

[![CI](https://github.com/gedouluhui/stepcontract/actions/workflows/ci.yml/badge.svg)](https://github.com/gedouluhui/stepcontract/actions/workflows/ci.yml)

**A parameter can have a gradient and still never reach your optimizer. Make the training intent explicit.**

StepContract checks a small, visible PyTorch training window: parameter ownership, declared freezing/accumulation intent, and the sequence of `zero_grad`, `backward` and `step`. It is a contract checker, not another training dashboard.

[中文说明](README.zh-CN.md)

## See a real mistake

From a checkout, using a Python environment with CPU PyTorch:

```sh
python -m pip install .
stepcontract demo --case healthy --html healthy.html
stepcontract demo --case missing-optimizer --html missing.html
stepcontract demo --case early-zero --html early-zero.html
```

Broken demos intentionally exit **1**. Open the HTML reports to see actual optimizer membership and event evidence. The demos run tiny CPU training operations; no model or dataset download and no API key.

## Use it in a training test

```python
import torch
from stepcontract import StepContract

model = torch.nn.Linear(2, 1, bias=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
audit = StepContract(model, optimizer, required_parameters=["weight"])

audit.zero_grad()
loss = model(torch.ones(1, 2)).sum()
audit.backward(loss)
audit.step()
report = audit.report()
assert report["status"] == "PASS", report["findings"]
```

The wrappers perform **only the operations you explicitly request**. They do not change learning rates, freeze parameters or clear gradients by themselves. There is no global monkeypatch and no persistent hook.

For gradient accumulation, specify `accumulation_steps=2`, call `backward` twice, then `step`. You remain responsible for scaling your loss correctly. Create a new audit object after intentionally changing parameter identities or optimizer membership.

## Interpret the evidence

- A **required** parameter missing from the optimizer or marked frozen violates declared intent. A nonrequired frozen parameter does not.
- Clearing gradients between backward and step records an event-order conflict. If a gradient disappears through an unobserved external operation, the report states the observed transition, not a guessed cause.
- `none`, `zero`, `nonzero`, `nonfinite` are separate gradient states. An unchanged weight is not automatically a bug: zero learning rate and conditional branches can be legitimate.
- Exact before/after comparisons distinguish changes that a norm-only check could miss. Aliases of the same Parameter are counted once.
- Reports contain names, shapes, ownership and observations, **not weight arrays, raw gradients or training examples**.

### Accepted calls, outcomes and provenance

Validation that happens **before** an event is admitted — a non-boolean `set_to_none`, the event-history cap, or an unsupported optimizer mode found at preflight — raises without starting an operation, without consuming history and without retroactively invalidating completed steps. Such a rejected call has no event index, and a historical PASS does not certify it.

Once a call is admitted, its event records `operation_started` and `operation_returned`, and `outcome` is `completed`, `failed` or `unsupported`. Every finding carries its `event_index` or `null` plus an `origin` (`construction`, `event` or `report`), so each claim resolves to a real observation instead of an invented index. If the exact pre-step snapshot cannot be taken, the optimizer is not called, `completed_steps` does not increase and the failure stays visible. After the first exception in an accepted event, the audit is **sticky ERROR**: a later successful retry cannot restore PASS.

| Status | Meaning |
|---|---|
| PASS | Completed visible window with no detected conflict against declared intent. |
| FAIL | Contract violation or nonfinite value observed. |
| INCOMPLETE | No completed step, or backward work still pending. |
| ERROR | An accepted wrapped operation failed. Its exception is re-raised to preserve caller error handling, and the report stays ERROR for the lifetime of the audit. |
| UNSUPPORTED | Unsupported observation mode. Constructor/preflight may raise `UnsupportedOperation`. |

CLI exit codes: PASS 0, FAIL 1, ERROR 2, INCOMPLETE 3, UNSUPPORTED 4.

## Scope, cost and non-goals

v0.1.1 supports **CPU dense contiguous FP32 parameters, single-process eager execution, unmodified SGD/AdamW**, with no optimizer closure. Differentiable/capturable/fused modes, AMP, distributed training, compiled graphs and separate Parameter objects sharing storage are outside scope.

Only calls through the audit object are tracked. Arbitrary external operations, optimizer hooks and concurrent training are not certified. This tool does not prove the optimizer's recurrence, model quality, convergence or whole-program correctness.

Snapshots cost O(P) memory and time. `max_snapshot_bytes` (default 64 MiB) checks parameter clone size before allocation; it is not a total process-RSS limit. Event history is capped. Construction and unsupported inputs can raise a clear exception rather than yielding a misleading PASS. Names and shapes can be sensitive; review reports before sharing.

## Alternatives and development

PyTorch anomaly detection targets backward errors and NaNs. Cockpit and SageMaker Debugger offer broader monitoring. StepContract focuses on **explicit caller intent and CI-friendly event evidence**, not a claim of globally new training diagnostics.

```sh
python -m unittest discover -s tests -v
python -m pip install build
python -m build --wheel
```

Tests use real SGD and AdamW controls and verify identical parameter/gradient updates with and without the auditor, legal freezing and accumulation, missing optimizer parameters, early/external clearing, nonfinite gradients, residue, identity changes, accepted-event failure and retry provenance, runtime optimizer-mode refusal and CLI reports. See [local evidence](docs/IMPLEMENTATION.md). No production training or external adoption claim is made.

Report bugs with a tiny synthetic training loop, not private model weights or datasets. AI-assisted implementation. MIT; PyTorch's own license remains applicable. This repository does not claim a PyPI release.

## Quickstart and support

### Problem
Make training-step intent explicit so a parameter with a gradient cannot silently miss the optimizer or violate the declared step sequence.

### Install
```sh
python -m pip install .
```

### Success
```sh
stepcontract demo --case healthy --html healthy.html
```
Expected: The healthy CPU training window completes with PASS and exit 0.

### Deliberate failure
```sh
stepcontract demo --case missing-optimizer --html missing.html
```
Expected: The missing-optimizer control reports a contract violation and exits 1.

### Evidence and coverage boundaries
Evidence uses real SGD and AdamW controls, identical-update checks, legal freezing and accumulation, missing optimizer parameters, early/external clearing, nonfinite gradients, identity changes, and CLI reports. Coverage is limited to observed calls through the audit object; arbitrary external operations and whole-program correctness are not certified.

### Support status
Version 0.1.1 supports Python 3.10+ with CPU PyTorch and the scoped eager FP32 SGD/AdamW workflow. The documented route is a source checkout; no PyPI release is claimed. Maintenance is best-effort with no SLA.

### Limitations
Distributed training, AMP, compiled graphs, optimizer closures, differentiable/capturable/fused modes, and separate Parameter objects sharing storage are outside scope. Snapshots cost O(P), and names/shapes can be sensitive.
