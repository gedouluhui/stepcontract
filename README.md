# StepContract

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

| Status | Meaning |
|---|---|
| PASS | Completed visible window with no detected conflict against declared intent. |
| FAIL | Contract violation or nonfinite value observed. |
| INCOMPLETE | No completed step, or backward work still pending. |
| ERROR | A wrapped operation failed. Its exception is re-raised to preserve caller error handling. |
| UNSUPPORTED | Unsupported observation mode. Constructor/preflight may raise `UnsupportedOperation`. |

CLI exit codes: PASS 0, FAIL 1, ERROR 2, INCOMPLETE 3, UNSUPPORTED 4.

## Scope, cost and non-goals

v0.1 supports **CPU dense contiguous FP32 parameters, single-process eager execution, unmodified SGD/AdamW**, with no optimizer closure. Differentiable/capturable/fused modes, AMP, distributed training, compiled graphs and separate Parameter objects sharing storage are outside scope.

Only calls through the audit object are tracked. Arbitrary external operations, optimizer hooks and concurrent training are not certified. This tool does not prove the optimizer's recurrence, model quality, convergence or whole-program correctness.

Snapshots cost O(P) memory and time. `max_snapshot_bytes` (default 64 MiB) checks parameter clone size before allocation; it is not a total process-RSS limit. Event history is capped. Construction and unsupported inputs can raise a clear exception rather than yielding a misleading PASS. Names and shapes can be sensitive; review reports before sharing.

## Alternatives and development

PyTorch anomaly detection targets backward errors and NaNs. Cockpit and SageMaker Debugger offer broader monitoring. StepContract focuses on **explicit caller intent and CI-friendly event evidence**, not a claim of globally new training diagnostics.

```sh
python -m unittest discover -s tests -v
python -m pip install build
python -m build --wheel
```

Tests use real SGD and AdamW controls and verify identical parameter/gradient updates with and without the auditor, legal freezing and accumulation, missing optimizer parameters, early/external clearing, nonfinite gradients, residue, identity changes and CLI reports. See [local evidence](docs/IMPLEMENTATION.md). No production training or external adoption claim is made.

Report bugs with a tiny synthetic training loop, not private model weights or datasets. AI-assisted implementation. MIT; PyTorch's own license remains applicable. This repository does not claim a PyPI release.
