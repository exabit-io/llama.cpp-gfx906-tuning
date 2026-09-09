#!/usr/bin/env python3
"""Renders reports/2026-09-09-fork-survey.md into reports/2026-09-09-fork-survey.html in the guide's report style
(the stylesheet of the 2026-09-07 report): headings, paragraphs, bullet lists, tables, bold, inline code, links."""
import re, html, datetime, os
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md = open(f'{G}/reports/2026-09-09-fork-survey.md').read()
css = open(f'{G}/reports/2026-09-07-todo-runthrough.html').read().split('<style>')[1].split('</style>')[0]
def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(https?://[^\s)<]+)', r'<a href="\1">\1</a>', t)
    return t
out = []; lines = md.splitlines(); i = 0; title = 'Fork Survey'
while i < len(lines):
    l = lines[i]
    if l.startswith('# '):
        title = l[2:]; out.append(f'<div class="eyebrow">gfx906 · Qwen3.8 · fork survey · generated {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</div><h1>{inline(l[2:])}</h1>'); i += 1
    elif l.startswith('## '):
        out.append(f'<h2>{inline(l[3:])}</h2>'); i += 1
    elif l.startswith('### '):
        out.append(f'<h3>{inline(l[4:])}</h3>'); i += 1
    elif l.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].startswith('|'):
            cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.match(r'^:?-+:?$', c) for c in cells): rows.append(cells)
            i += 1
        t = ['<div class="tablewrap"><table><thead><tr>' + ''.join(f'<th>{inline(c)}</th>' for c in rows[0]) + '</tr></thead><tbody>']
        for r in rows[1:]:
            t.append('<tr>' + ''.join(f'<td class="{"num" if re.match(r"^[-+±0-9.,/ %()xk~]+$", c) else ""}">{inline(c)}</td>' for c in r) + '</tr>')
        out.append('\n'.join(t) + '</tbody></table></div>')
    elif l.startswith('- '):
        items = []
        while i < len(lines) and lines[i].startswith('- '):
            items.append(f'<li>{inline(lines[i][2:])}</li>'); i += 1
        out.append('<ul>' + ''.join(items) + '</ul>')
    elif l.strip() == '':
        i += 1
    else:
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||- )', lines[i]):
            para.append(lines[i]); i += 1
        out.append(f'<p>{inline(" ".join(para))}</p>')
page = f'''<title>gfx906 Fork Survey</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@500;700;800&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}
.tablewrap{{overflow-x:auto;margin:12px 0}} td.num{{text-align:right;font-variant-numeric:tabular-nums}}</style>
<div class="wrap">
{chr(10).join(out)}
<p class="small">Source: <code>reports/2026-09-09-fork-survey.md</code>, rendered by <code>tools/gen-night-report.py</code>.</p>
</div>'''
open(f'{G}/reports/2026-09-09-fork-survey.html', 'w').write(page)
print('written', len(page))
