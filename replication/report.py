"""Build the replication report (report.html and summary.json) from one run folder of run_bench.py.

Usage: .venv/bin/python replication/report.py replication/results/<run name>
run_bench.py calls build() at the end of every run, so this is needed only to rebuild a report.

The estimator is that of the CommoditizeTeraModel study (EXP609), applied to the new GPU's serves. The design of
the coding pass differs from the study's coding run (EXP605); see the limits in the report.

  ms/token per prompt   the client-side RAW-clock span from the first to the last streamed piece over decode - 1 tokens
                        (decode only, prefill excluded). Read again from the request files, not from the record.
  blocks                the adjacent timed pairs (R584_1 - F0Q_1), (R584_2 - F0Q_2), (R584_3 - F0Q_3), per prompt.
  pooled                the mean of the three block differences, per prompt. Negative means R584 is faster.
  interval              99 % prompt bootstrap: 2000 draws of the prompts with replacement, one fresh draw matrix per
                        group from the study's seed (code 605, math and qa 607). Conditional on the observed serves.
  verdict               'faster' if the interval is below 0 and all three block means are below 0; 'slower' if the
                        mirror holds; otherwise 'unresolved'.
  serve level           Student t half-width of the serve means (two arms, three serves each, df 4). The label keeps the
                        verdict only if the pooled difference is larger than this half-width, so a result must stand
                        above serve-to-serve drift as well as prompt-to-prompt spread.
  tokens/s              per arm, 1000 / (mean over prompts of the per-prompt mean over the arm's timed serves). The rate
                        interval resamples prompts with the same draw matrix (the study's rates_v3 method).

A smoke run (CONFIG['max_scored_rows'] set) gets the same tables with a banner: a few prompts give wide intervals and
are not a result.
"""
from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / 'reference_rtx4080.json'
SEEDS = {'code': 605, 'math': 607, 'qa': 607}   # the study's estimator seeds (EXP605 and EXP609)
DRAWS = 2000
QUANTILES = (0.005, 0.995)
BLOCKS = (('R584_1', 'F0Q_1'), ('R584_2', 'F0Q_2'), ('R584_3', 'F0Q_3'))
# Student t 0.995 quantiles by degrees of freedom: df 4 = two arms of three serves (the registered design), df 2 = two
# arms of two serves (only if CONFIG['serves'] is edited to two per arm).
T_995 = {2: 9.924843, 4: 4.604095}
GROUP_TITLE = {'code': 'Coding, 1024 tokens', 'math': 'Math (MATH-500), 256 tokens', 'qa': 'Open questions (NQ-open), 256 tokens'}


class ReportError(Exception):
    pass


def need(ok: bool, what: str) -> None:
    if not ok:
        raise ReportError(what)


# ---- statistics (pure functions; test_replication.py checks them against the study's EXP609 estimate) ----------------

def draws(m: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, m, (DRAWS, m))


def interval(x: np.ndarray, idx: np.ndarray) -> list[float]:
    lo, hi = np.quantile(x[idx].mean(axis=1), QUANTILES)
    return [float(lo), float(hi)]


def verdict(ci: list[float], blocks: list[float]) -> str:
    if ci[0] > 0 and all(b > 0 for b in blocks):
        return 'slower'
    if ci[1] < 0 and all(b < 0 for b in blocks):
        return 'faster'
    return 'unresolved'


def serve_level(table: dict[str, np.ndarray]) -> dict:
    arms: dict[str, list[float]] = {}
    for name, v in table.items():
        arms.setdefault(name.rsplit('_', 1)[0], []).append(float(v.mean()))
    counts = {len(m) for m in arms.values()}
    need(len(arms) == 2 and len(counts) == 1, f'serve level needs two arms with equal serve counts: {arms.keys()}')
    n = counts.pop()
    df = 2 * (n - 1)
    need(df in T_995, f'no t quantile for df {df}')
    sd2 = sum(float(np.var(m, ddof=1)) for m in arms.values()) / 2
    return dict(serve_means_ms=arms, df=df, half_width_ms=T_995[df] * math.sqrt(sd2 * 2 / n))


def contrast_ms(table: dict[str, np.ndarray], seed: int) -> dict:
    """R584 - F0Q in ms/token over one group's prompts (the study's EXP609 contrast)."""
    diffs = [table[x] - table[y] for x, y in BLOCKS]
    m = len(diffs[0])
    need(m >= 2 and all(d.shape == (m,) and np.isfinite(d).all() for d in diffs), 'blocks need >= 2 finite values')
    pooled = sum(diffs) / len(diffs)
    idx = draws(m, seed)
    ci = interval(pooled, idx)
    blocks = [float(d.mean()) for d in diffs]
    level = serve_level(table)
    point = float(pooled.mean())
    v = verdict(ci, blocks)
    return dict(pooled_ms=point, pooled_ci_99=ci, block_ms=blocks, verdict=v,
                label=v if v != 'unresolved' and abs(point) > level['half_width_ms'] else 'unresolved',
                half_width_ms=level['half_width_ms'], serve_means_ms=level['serve_means_ms'], prompts=m)


def contrast_rate(table: dict[str, np.ndarray], seed: int) -> dict:
    """Tokens/s of R584 (x) and F0Q (y) and their difference and percent gain, with the same prompt draws."""
    def arm(a):
        return np.mean([v for k, v in table.items() if k.rsplit('_', 1)[0] == a], axis=0)
    x, y = arm('R584'), arm('F0Q')
    idx = draws(len(x), seed)
    rx, ry = 1000 / x.mean(), 1000 / y.mean()
    bx, by = 1000 / x[idx].mean(axis=1), 1000 / y[idx].mean(axis=1)
    return dict(x_tok_s=float(rx), y_tok_s=float(ry), diff_tok_s=float(rx - ry),
                diff_ci_99=[float(v) for v in np.quantile(bx - by, QUANTILES)],
                gain_pct=float(100 * (rx / ry - 1)),
                gain_pct_ci_99=[float(v) for v in np.quantile(100 * (bx / by - 1), QUANTILES)],
                prompts=len(x), prompts_x_faster=int((x < y).sum()))


# ---- reading a run ---------------------------------------------------------------------------------------------------

def ms_per_token(q: dict) -> float:
    s = q['stamps_raw']
    need(len(s) == q['events'] and 2 <= len(s) <= q['decode'] and all(b >= a for a, b in zip(s, s[1:])) and s[-1] > s[0],
         f"request {q.get('number')}: invalid stamps")
    return 1000 * (s[-1] - s[0]) / (q['decode'] - 1)


def group_table(pass_dir: Path, record: dict, rows: list[int]) -> dict[str, np.ndarray]:
    """Per timed serve, the ms/token of the given rows, read again from the request files. Each value must equal the
    serve's gated record, so a hand-edited request file or record cannot pass unnoticed."""
    table = {}
    for x, y in BLOCKS:
        for name in (x, y):
            serve = record['serves'][name]
            vals = []
            for n in rows:
                q = json.loads((pass_dir / f'serve_{name}' / f'request_{n:02d}.json').read_text())
                need(q.get('number') == n, f'{name}/request_{n:02d}.json has another number')
                v = ms_per_token(q)
                need(v == serve['ms_per_token_by_prompt'][str(n)], f'{name} row {n}: request file differs from record')
                vals.append(v)
            table[name] = np.asarray(vals, dtype=np.float64)
    return table


def analyse(run_dir: Path) -> dict:
    info = json.loads((run_dir / 'run.json').read_text())
    out = dict(run=info, groups={}, passes={})
    for p in info.get('config', {}).get('passes', []):
        rec_path = run_dir / p / 'record.json'
        if not rec_path.is_file():
            continue
        record = json.loads(rec_path.read_text())
        out['passes'][p] = dict(status=record.get('status'), error=record.get('error'),
                                cross_gates=record.get('cross_gates'), equal_memory=record.get('equal_memory'),
                                serves={n: dict(passed=s.get('passed'), startup_s=s.get('startup_s'),
                                                serve_s=s.get('serve_s'), device_use_bytes=s.get('device_use_bytes'),
                                                free_after_init_gib=s.get('free_after_init_gib'),
                                                free_after_capture_gib=s.get('free_after_capture_gib'),
                                                # The study's rule: timings of a failed pass are not shown.
                                                mean_ms=s.get('ms_per_token_mean')
                                                if record.get('status') == 'complete' else None,
                                                pcie=s.get('pcie'), gpu_watch=s.get('gpu_watch'))
                                        for n, s in record.get('serves', {}).items()})
        if record.get('status') != 'complete':
            continue
        for g, rows in record['groups'].items():
            table = group_table(run_dir / p, record, rows)
            out['groups'][g] = dict(pass_name=p, decode=record['decode'][str(rows[0])],
                                    ms=contrast_ms(table, SEEDS[g]), rate=contrast_rate(table, SEEDS[g]))
    return out


# ---- HTML ------------------------------------------------------------------------------------------------------------

def esc(v) -> str:
    return '–' if v is None else html.escape(str(v))


def fmt_ci(ci, nd=1) -> str:
    return f'[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]'


def bar_chart(groups: list[str], ref: dict, new: dict, gpu: str) -> str:
    """Grouped bars: tokens/s of F0Q and R584 on the RTX 4080 and on the new GPU, one cluster per group."""
    w, h, left, bottom, top = 760, 320, 56, 56, 24
    series = [('RTX 4080 · F0Q', 'ref', 'y_tok_s', 'var(--c1l)'), ('RTX 4080 · R584', 'ref', 'x_tok_s', 'var(--c1)'),
              (f'{gpu} · F0Q', 'new', 'y_tok_s', 'var(--c2l)'), (f'{gpu} · R584', 'new', 'x_tok_s', 'var(--c2)')]
    vals = [(ref if s == 'ref' else new).get(g, {}).get(k) for g in groups for _, s, k, _ in series]
    vmax = max([v for v in vals if v is not None] + [1.0])
    step = 20 if vmax <= 200 else 50
    ymax = step * math.ceil(vmax * 1.08 / step)
    plot_w, plot_h = w - left - 16, h - bottom - top
    cw = plot_w / len(groups)
    bw = min(34, (cw - 30) / 4 - 2)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Decode speed in tokens per second by arm, GPU and test group">']
    for t in range(0, int(ymax) + 1, step):
        y = top + plot_h * (1 - t / ymax)
        parts.append(f'<line x1="{left}" x2="{w - 16}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" class="tick">{t}</text>')
    for gi, g in enumerate(groups):
        x0 = left + gi * cw + (cw - 4 * (bw + 2)) / 2
        for si, (name, s, k, color) in enumerate(series):
            v = (ref if s == 'ref' else new).get(g, {}).get(k)
            x = x0 + si * (bw + 2)
            if v is None:
                parts.append(f'<text x="{x + bw / 2:.1f}" y="{top + plot_h - 4}" text-anchor="middle" class="tick">–</text>')
                continue
            bh = plot_h * v / ymax
            parts.append(f'<rect x="{x:.1f}" y="{top + plot_h - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" '
                         f'fill="{color}"><title>{esc(name)}, {esc(g)}: {v:.2f} tokens/s</title></rect>'
                         f'<text x="{x + bw / 2:.1f}" y="{top + plot_h - bh - 4:.1f}" text-anchor="middle" '
                         f'class="val">{v:.0f}</text>')
        parts.append(f'<text x="{left + gi * cw + cw / 2:.1f}" y="{top + plot_h + 18}" text-anchor="middle" '
                     f'class="lab">{esc(GROUP_TITLE.get(g, g))}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{h - 8}" text-anchor="middle" class="axis">Test group (x axis)</text>'
                 f'<text transform="translate(14 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" class="axis">'
                 f'Decode speed, tokens/s (y axis)</text></svg>')
    legend = ''.join(f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, _, _, c in series)
    return f'<figure>{"".join(parts)}<figcaption><div class="legend">{legend}</div>Bars are decode speed in tokens per ' \
           f'second (prefill excluded), the mean over the prompts of each arm\'s timed serves. Higher is better.</figcaption></figure>'


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1f23;--muted:#5d6470;--line:#dfe2e6;--card:#fff;--c1:#2f6fd0;--c1l:#a9c4ee;--c2:#c2571a;--c2l:#efbf9f;--ok:#1b7f3b;--bad:#b42318}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171a;--fg:#e8eaed;--muted:#a2a9b4;--line:#30343a;--card:#1d2024;--c1:#6b9ff0;--c1l:#35507a;--c2:#ee8a4f;--c2l:#7a4527;--ok:#5fcf84;--bad:#ff7b6b}}
:root[data-theme="dark"]{--bg:#15171a;--fg:#e8eaed;--muted:#a2a9b4;--line:#30343a;--card:#1d2024;--c1:#6b9ff0;--c1l:#35507a;--c2:#ee8a4f;--c2l:#7a4527;--ok:#5fcf84;--bad:#ff7b6b}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}
main{max-width:860px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px;border-top:1px solid var(--line);padding-top:20px}
.sub{color:var(--muted)}.banner{border:1px solid var(--line);border-left:4px solid var(--c2);background:var(--card);padding:10px 14px;margin:16px 0}
.state-A{border-left-color:var(--ok)}.state-B,.state-C,.state-interrupted{border-left-color:var(--bad)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
td.num,th{white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:12px 0}svg{max-width:100%;height:auto;display:block}
.grid{stroke:var(--line)}.tick,.val{fill:var(--muted);font-size:11px}.lab{fill:var(--fg);font-size:12px}.axis{fill:var(--muted);font-size:12px}
figcaption{color:var(--muted);font-size:13px}.legend{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0;color:var(--fg)}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.pass{color:var(--ok)}.fail{color:var(--bad)}dt{font-weight:600}dd{margin:0 0 6px 0;color:var(--muted)}
h3{font-size:15px;margin:14px 0 4px}
"""


def section_end(means: str, conclude: str) -> str:
    return f'<h3>What this means</h3><p>{means}</p><h3>What we can conclude</h3><p>{conclude}</p>'


def render(a: dict, ref: dict) -> str:
    info = a['run']
    gpu = (info.get('hardware') or {}).get('gpus', [{}])[0].get('name', 'new GPU')
    state = info.get('state')
    gpu_short = gpu.replace('NVIDIA GeForce ', '')
    groups = [g for g in ('code', 'math', 'qa') if g in a['groups']]
    new_rates = {g: a['groups'][g]['rate'] for g in groups}
    ref_rates = {g: ref['groups'][g]['rate'] for g in ('code', 'math', 'qa')}
    state_text = {'A': 'A: complete. Every gate passed.', 'B': 'B: stopped by a gate or an error after the start.',
                  'C': 'C: did not start. A preflight check failed; nothing was timed.',
                  'interrupted': 'Interrupted with Ctrl-C.'}.get(state, f'unknown ({state})')
    h = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
         f'initial-scale=1"><title>CommitMoE Replication Report</title><style>{CSS}</style></head><body><main>',
         f'<h1>CommitMoE replication on {esc(gpu_short)}</h1>',
         f'<p class="sub">Run {esc(info.get("run"))}, {esc(info.get("start_wall"))} to {esc(info.get("end_wall"))}. '
         f'Engine commit {esc(info.get("engine_commit", "")[:12])}.</p>',
         f'<div class="banner state-{esc(state)}"><b>End state {state_text}</b>'
         + (f'<br>{esc(info.get("failure"))}' if info.get('failure') else '') + '</div>']
    if info.get('smoke'):
        h.append(f'<div class="banner"><b>Smoke run:</b> only the first {esc(info["config"]["max_scored_rows"])} prompts '
                 f'of each group ran. The intervals are wide and the numbers are not a result.</div>')

    # glossary
    h.append('<h2>Glossary</h2><dl>'
             '<dt>F0Q</dt><dd>FreeToken offload with striped demand copies and no predictor, 3571 expert slots.</dd>'
             '<dt>R584</dt><dd>The same engine with the CommitMoE predictor (copies experts early), copy overlap and the '
             'learned keep-for-later eviction, 3496 expert slots. On the RTX 4080 the 75 fewer slots paid for the '
             'predictor, so both arms used the same GPU memory.</dd>'
             '<dt>Expert slot</dt><dd>GPU memory for one expert (3146112 bytes). The rest of the 10240 experts stay in host RAM '
             'and cross PCIe on a miss.</dd>'
             '<dt>Decode speed</dt><dd>1000 divided by the mean milliseconds per token. Milliseconds per token is the client '
             'time from the first to the last streamed text piece, divided by the generated tokens minus one. A piece '
             'need not be exactly one token, so this is a proxy for the decode rate, not an exact token count over '
             'time. Both arms stream the same pieces (the text gate), so the two arms measure the same span. Prefill is '
             'excluded.</dd>'
             '<dt>Serve</dt><dd>One server start. Each pass starts 7: one untimed R584 warm-up, then F0Q_1, R584_1, R584_2, '
             'F0Q_2, F0Q_3, R584_3.</dd>'
             '<dt>Block</dt><dd>An adjacent timed pair (F0Q_1, R584_1), (R584_2, F0Q_2), (F0Q_3, R584_3).</dd>'
             '<dt>99 % interval</dt><dd>Prompt bootstrap, 2000 draws. It shows the spread over prompts, conditional on '
             'these serves. It is not an interval over machines.</dd>'
             '<dt>EXP605, EXP609</dt><dd>The RTX 4080 experiments that this run repeats (coding at 1024 tokens; math and '
             'open questions at 256 tokens).</dd></dl>')

    # hardware
    hw, sw = info.get('hardware') or {}, info.get('software') or {}
    g0 = (hw.get('gpus') or [{}])[0]
    pcie = next((s.get('pcie') for p in a['passes'].values() for s in p['serves'].values() if s.get('pcie')), None)
    rh = ref['hardware']
    rows = [('GPU', rh['gpu'], g0.get('name')), ('GPU memory, MiB', rh['vram_mib'], g0.get('memory_total_mib')),
            ('PCIe link (max)', rh['pcie_max'], f"gen {g0.get('pcie_gen_max')} x{g0.get('pcie_width_max')}"),
            ('PCIe link under load', '–', f"gen {pcie['gen_current_max']} x{pcie['width_current_max']} "
                                          f"({pcie['samples']} samples)" if pcie else 'not sampled'),
            ('Driver', '–', g0.get('driver')), ('Host', rh['host'], f"{hw.get('cpu')}, "
                                                  f"{(hw.get('mem_total_bytes') or 0) / 2 ** 30:.0f} GiB RAM"),
            ('OS', 'WSL2 on Windows 11', f"{hw.get('os')} ({'WSL2' if hw.get('wsl') else 'native Linux'}), kernel "
                                         f"{hw.get('kernel')}"),
            ('torch / CUDA / triton', '2.14.1 / 13 / –', f"{sw.get('torch')} / {sw.get('cuda')} / {sw.get('triton')}"),
            ('Compute capability', '8.9', '.'.join(str(c) for c in sw.get('capability', [])) or '–')]
    h.append('<h2>Machines</h2><table><tr><th></th><th>RTX 4080 (study)</th><th>This run</th></tr>'
             + ''.join(f'<tr><td>{esc(k)}</td><td>{esc(r)}</td><td>{esc(n)}</td></tr>' for k, r, n in rows) + '</table>'
             + section_end('Both runs used the same engine commit, flags, slot counts, KV pages and prompts. The GPU, the '
                           'PCIe link, the host CPU and RAM, and the operating system differ.',
                           'A difference in the results below can come from the machine or from run-to-run noise, not from '
                           'the method or its settings. This run alone cannot say which part of the machine caused a difference.'))

    # results
    if groups:
        h.append('<h2>Decode speed</h2>' + bar_chart(groups, ref_rates, new_rates, gpu_short))
        h.append('<table><tr><th>Group</th><th>GPU</th><th>F0Q tok/s</th><th>R584 tok/s</th><th>Gain, tok/s [99 %]</th>'
                 '<th>Gain, % [99 %]</th><th>R584 faster on</th><th>Label (ms test)</th></tr>')
        for g in groups:
            for who, rate, ms in (('RTX 4080', ref_rates[g], ref['groups'][g]['ms']),
                                  (gpu_short, new_rates[g], a['groups'][g]['ms'])):
                h.append(f'<tr><td>{esc(GROUP_TITLE[g])}</td><td>{esc(who)}</td><td class="num">{rate["y_tok_s"]:.2f}</td>'
                         f'<td class="num">{rate["x_tok_s"]:.2f}</td><td class="num">{rate["diff_tok_s"]:+.2f} '
                         f'{fmt_ci(rate["diff_ci_99"], 2)}</td><td class="num">{rate["gain_pct"]:+.1f} '
                         f'{fmt_ci(rate["gain_pct_ci_99"])}</td><td class="num">{rate["prompts_x_faster"]} of '
                         f'{rate["prompts"]}</td><td>{esc(ms["label"])}</td></tr>')
        h.append('</table>')
        lines = []
        for g in groups:
            r, n = ref_rates[g], new_rates[g]
            lines.append(f'{GROUP_TITLE[g]}: on the {esc(gpu_short)}, R584 decoded at {n["x_tok_s"]:.2f} tokens/s and F0Q '
                         f'at {n["y_tok_s"]:.2f}, a gain of {n["gain_pct"]:+.1f} % (99 % interval '
                         f'{fmt_ci(n["gain_pct_ci_99"])}). R584 was faster on {n["prompts_x_faster"]} of {n["prompts"]} '
                         f'prompts. The RTX 4080 gain was {r["gain_pct"]:+.1f} % {fmt_ci(r["gain_pct_ci_99"])}.')
        labels = {g: a['groups'][g]['ms']['label'] for g in groups}
        resolved = [g for g in groups if labels[g] == 'faster']
        concl = (f'R584 is faster than F0Q on this GPU in {len(resolved)} of {len(groups)} groups by the study\'s '
                 f'pre-set rule (interval and all three blocks below zero, and the effect larger than the serve-to-serve '
                 f'drift). ' if not info.get('smoke') else 'This is a smoke run; no conclusion about speed. ')
        concl += ('The intervals cover prompts only. With one machine, the run does not say how other RTX 5090 hosts '
                  'would behave.')
        h.append(section_end('<br>'.join(lines), concl))

    # gates
    h.append('<h2>Gates</h2>')
    for p, d in a['passes'].items():
        mem = d.get('equal_memory') or {}
        h.append(f'<h3>Pass {esc(p)}: {esc(d["status"])}</h3>' + (f'<p class="fail">{esc(d["error"])}</p>' if d.get('error') else ''))
        h.append('<table><tr><th>Serve</th><th>Passed</th><th>Start, s</th><th>Serve, s</th><th>Free after init / '
                 'capture, GiB</th><th>Device use, GiB</th><th>Mean tok/s</th></tr>')
        for n, s in d['serves'].items():
            du = s.get('device_use_bytes')
            ok = 'pass' if s['passed'] else 'fail'
            h.append(f'<tr><td>{esc(n)}</td><td class="{ok}">{"yes" if s["passed"] else "no"}</td>'
                     f'<td class="num">{esc(s.get("startup_s"))}</td><td class="num">{esc(s.get("serve_s"))}</td>'
                     f'<td class="num">{esc(s.get("free_after_init_gib"))} / {esc(s.get("free_after_capture_gib"))}</td>'
                     f'<td class="num">{"–" if du is None else f"{du / 2 ** 30:.2f}"}</td>')
            h.append(f'<td class="num">{1000 / s["mean_ms"]:.2f}</td></tr>' if s.get('mean_ms') else '<td>–</td></tr>')
        h.append('</table>')
        cross = d.get('cross_gates') or {}
        watched = [s.get('gpu_watch') for s in d['serves'].values() if s.get('gpu_watch')]
        if watched:
            seen = sum(1 for w in watched if w.get('own_seen'))
            errors = sum(w.get('errors', 0) for w in watched)
            # 'Not visible' is claimed only for a serve whose queries all succeeded and listed no process at all.
            blind = sum(1 for w in watched if not w.get('own_seen') and not w.get('errors') and w.get('samples')
                        and w.get('empty') == w.get('samples'))
            h.append(f'<p>GPU watch: {sum(w.get("samples", 0) for w in watched)} successful samples and {errors} '
                     f'failed queries in {len(watched)} serves; {sum(1 for w in watched if w.get("ok"))} serves met '
                     f'the coverage rule (at least one sample, no failed query, thread ended). The server\'s own '
                     f'process was visible in {seen} of {len(watched)} serves.'
                     + (f' In {blind} serves every query succeeded and listed no process, not even the server\'s '
                        'own: there the watch could not detect another process.' if blind else '') + '</p>')
        h.append(f'<p>Output identity and miss checks: {sum(1 for v in cross.values() if v)} of {len(cross)} passed. '
                 + (f'Device use, mean F0Q minus mean R584: {mem["difference_slots"]:+.2f} slots (allowed −5 to +2).'
                    if mem else '') + '</p>')
    h.append(section_end('Every serve passed the study\'s gates before its first request and after its last one: the '
                         'registered flags in the live engine, enough free GPU memory, CUDA graphs on, the predictor '
                         'loaded only in R584, and complete, correctly sized answers. Every serve streamed the same text '
                         'as the R584 warm-up serve (greedy decoding).' if state == 'A' else
                         'The run stopped at the first failed gate. The tables show the serves that ran.',
                         'With all gates passed, both arms ran the registered method on equal GPU memory and gave the same '
                         'output text. The speed numbers above compare like with like.' if state == 'A' else
                         'Fix the reason shown in the banner and start a new run. Do not mix serves of two runs.'))

    h.append('<h2>Limits</h2><ul>'
             '<li>One machine, batch 1, greedy decoding. The intervals cover prompts, not machines or days.</li>'
             '<li>The slot counts are those of the 16 GB RTX 4080. A 32 GB GPU could hold more experts; this run does not '
             'test that configuration on purpose.</li>'
             '<li>The text gate compares serves of this run. It does not compare with the RTX 4080 text, because a '
             'different GPU can change the low bits of the arithmetic.</li>'
             '<li>The RTX 4080 ran the engine in WSL2. A native Linux host can copy faster over PCIe.</li>'
             '<li>Nobody tested the engine\'s Triton kernels on this GPU before this run.</li>'
             '<li>The coding pass uses three timed serves per arm and the same estimator as math and open questions. '
             'The study\'s coding run (EXP605) used two timed serves per arm and estimated serve-to-serve drift from '
             'four arms. The coding numbers therefore compare in direction and size with EXP605, not as a repeat of '
             'the same statistical design.</li>'
             '<li>The GPU watch sees other processes only where the driver shows process ids. Where it never saw the '
             'server itself (see the gates), it could not see another process either.</li>'
             '<li>The model shards were not hashed. The model is pinned by its download revision.</li></ul>')
    h.append('</main></body></html>')
    return '\n'.join(h)


def build(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    ref = json.loads(REFERENCE.read_text())
    a = analyse(run_dir)
    (run_dir / 'summary.json').write_text(json.dumps(dict(groups=a['groups'], passes=a['passes'],
                                                          state=a['run'].get('state')), indent=1, default=str) + '\n')
    page = run_dir / 'report.html'
    page.write_text(render(a, ref))
    return page


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: report.py <run folder>')
    print(build(Path(sys.argv[1])))
