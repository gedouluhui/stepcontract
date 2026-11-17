import argparse
import json
import sys
import torch

from .audit import StepContract
from .contract import UnsupportedOperation
from .report import write_reports

CODES = {'PASS': 0, 'FAIL': 1, 'ERROR': 2, 'INCOMPLETE': 3, 'UNSUPPORTED': 4}


def demo(case):
    m = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.tensor([[1., 2.]]))
    parameters = [torch.nn.Parameter(torch.ones(1))] if case == 'missing-optimizer' else m.parameters()
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    count = 2 if case == 'accumulation' else 1
    audit = StepContract(m, optimizer, required_parameters=['weight'], accumulation_steps=count)
    audit.zero_grad()
    for _ in range(count):
        x = torch.tensor([[float('nan'), 1.]]) if case == 'nonfinite' else torch.ones(1, 2)
        audit.backward(m(x).sum())
    if case == 'early-zero':
        audit.zero_grad()
    audit.step()
    return audit.report()


def main(argv=None):
    p = argparse.ArgumentParser(description='Check explicit PyTorch training intent against visible step events.')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('demo', help='run a tiny real CPU training scenario')
    d.add_argument('--case', choices=['healthy', 'missing-optimizer', 'early-zero', 'nonfinite', 'accumulation'], default='healthy')
    d.add_argument('--json')
    d.add_argument('--html')
    args = p.parse_args(argv)
    try:
        report = demo(args.case)
        print(write_reports(report, args.json, args.html))
        return CODES[report['status']]
    except UnsupportedOperation as exc:
        print(json.dumps({'status': 'UNSUPPORTED', 'error': str(exc)}))
        return 4
    except (ValueError, OSError) as exc:
        print(json.dumps({'status': 'ERROR', 'error': str(exc) if isinstance(exc, ValueError) else type(exc).__name__}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
