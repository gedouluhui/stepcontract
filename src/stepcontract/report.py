"""Local-only, escaped event and ownership evidence."""
from html import escape
import json
import os
from pathlib import Path
import tempfile


def write_reports(report, json_path=None, html_path=None):
    paths = [Path(p) for p in (json_path, html_path) if p is not None]
    if len({p.resolve() for p in paths}) != len(paths) or any(p.is_symlink() for p in paths):
        raise ValueError('report paths must be distinct, non-symlink files')
    if len(paths) == 2 and all(p.exists() for p in paths) and os.path.samefile(*paths):
        raise ValueError('report paths must not be hardlink aliases')
    if any(not p.parent.is_dir() or (p.exists() and not p.is_file()) for p in paths):
        raise ValueError('report paths need existing parent directories')
    text = json.dumps(report, indent=2, allow_nan=False, ensure_ascii=True)
    for path, content in ((json_path, text), (html_path, to_html(report))):
        if path is None:
            continue
        path = Path(path)
        fd, temporary = tempfile.mkstemp(prefix='.stepcontract-', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
                stream.write(content)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return text


def to_html(report):
    e = lambda x: escape(str(x), quote=True)
    events = ''.join('<li><span>' + e(item['index']) + '</span><strong>' + e(item['kind']) + '</strong></li>' for item in report['events'])
    rows = ''.join('<tr><td><code>' + e(', '.join(p['names'])) + '</code></td><td>' + e(p['optimizer_memberships']) +
                   '</td><td>' + e(p['gradient']) + '</td><td>' + e(p['changed_last_step']) + '</td></tr>' for p in report['parameters'])
    findings = ''.join('<article><b>' + e(f['rule']) + '</b><code>' + e(f['parameter']) + '</code><p>' + e(f['detail']) + '</p></article>' for f in report['findings'])
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>StepContract · Training evidence</title><style>*{box-sizing:border-box}body{margin:0;background:#f1f5f7;color:#162536;font:16px/1.65 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:48px 24px}header{border-bottom:2px solid #006e75;padding-bottom:22px}.brand{color:#006e75;font:600 14px ui-monospace,monospace;letter-spacing:.12em}h1{font:500 clamp(32px,6vw,58px)/1.15 Georgia,serif;margin:22px 0}.status{background:#162536;color:white;padding:8px 16px;display:inline-block}h2{margin-top:36px}ol{display:flex;flex-wrap:wrap;gap:10px;list-style:none;padding:0}li{display:flex;gap:12px;align-items:center;padding:12px 18px;background:#e1edef;border-bottom:3px solid #006e75}li span{font:14px ui-monospace,monospace;color:#006e75}table{width:100%;border-collapse:collapse;background:white}.scroll{overflow:auto}th,td{padding:12px;text-align:left;border-bottom:1px solid #cedce1}th{font-size:13px}code{font:13px/1.6 ui-monospace,monospace;overflow-wrap:anywhere}article{border-left:3px solid #aa2e43;background:white;padding:16px 20px;margin:12px 0}article code{display:block}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#e1edef;padding:20px;font-size:12px}summary{cursor:pointer}summary:focus-visible{outline:3px solid #006e75}footer{margin-top:32px;font-size:13px;color:#445e70}@media(max-width:600px){main{padding:24px 14px}}</style><main><header><div class="brand">STEPCONTRACT / VISIBLE TRAINING EVENTS</div><h1>Intent. Gradients.<br>Actual updates.</h1><span class="status">''' + e(report['status']) + '''</span></header><p>''' + e(report['scope']) + '''</p><h2>The observed sequence</h2><ol>''' + events + '''</ol><h2>Parameter ownership</h2><div class="scroll"><table><tr><th>Names / aliases</th><th>Optimizer entries</th><th>Gradient</th><th>Changed this step</th></tr>''' + rows + '''</table></div><h2>Contract findings</h2>''' + (findings or '<p>No conflict with the declared intent was observed.</p>') + '''<details><summary>Complete machine-readable evidence</summary><pre>''' + e(json.dumps(report, indent=2, allow_nan=False)) + '''</pre></details><footer>Weights, raw gradients and training data are not embedded. An unchanged parameter is an observation, not automatically a bug.</footer></main></html>'''
