#!/usr/bin/env python3
"""Render the canonical research source as a self-contained, readable HTML report."""
from pathlib import Path
import re
import markdown

root = Path(__file__).resolve().parent
source = (root / "report-source.md").read_text()
md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"], extension_configs={"toc": {"toc_depth": "2"}})
body = md.convert(source)
body = re.sub(r"<table>", '<div class="table-scroll" role="region" aria-label="Roadmap comparison" tabindex="0"><table>', body)
body = body.replace("</table>", "</table></div>")
css = """
:root{color-scheme:light;--ink:#19343d;--muted:#52636b;--accent:#007b79;--line:#d7e4e6;--paper:#fff}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:#f3f6f6;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:17px;line-height:1.65}
.mast{background:#153943;color:#d8efed;padding:24px max(24px,calc((100vw - 1160px)/2));font-size:13px;letter-spacing:.13em;text-transform:uppercase}
.layout{display:grid;grid-template-columns:235px minmax(0,875px);gap:40px;max-width:1200px;margin:44px auto;padding:0 24px 64px}
aside{position:sticky;top:24px;align-self:start;font-size:14px;line-height:1.45}aside h2{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}aside ul{padding:0;list-style:none}aside li{margin:0 0 12px}aside a{color:var(--muted);text-decoration:none}aside a:hover{color:var(--accent);text-decoration:underline}
main{background:var(--paper);padding:46px 52px;min-width:0;border:1px solid var(--line);border-radius:8px;box-shadow:0 4px 28px #163d4510}h1{font-size:42px;line-height:1.12;letter-spacing:-.035em;margin:0 0 20px;color:#143c43}h1+p{font-size:14px;color:var(--muted);padding-bottom:24px;border-bottom:1px solid var(--line)}h2{font-size:26px;line-height:1.25;margin:48px 0 20px;letter-spacing:-.02em;scroll-margin-top:24px}h3{font-size:20px;line-height:1.35;margin:34px 0 12px;scroll-margin-top:24px}p{margin:0 0 18px}a{color:#006d77;text-decoration-thickness:1px;text-underline-offset:3px;overflow-wrap:anywhere}strong{font-weight:650}code{font-size:.85em;background:#eef3f4;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}pre{background:#19343d;color:#eef8f8;padding:20px;border-radius:6px;overflow:auto;line-height:1.5}pre code{background:none;padding:0;white-space:pre;overflow-wrap:normal}
.table-scroll{overflow:auto;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5}th{text-align:left;color:#fff;background:#1e555d;padding:12px;min-width:140px}td{vertical-align:top;border-bottom:1px solid var(--line);padding:12px}tr:nth-child(even){background:#f5f9f9}ol{padding-left:24px}li{padding-left:3px;margin-bottom:12px}.footer{margin-top:34px;padding-top:20px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
@media(max-width:950px){.layout{display:block;max-width:860px;margin-top:20px}aside{position:static;padding:0 12px 20px}aside ul{display:flex;flex-wrap:wrap;gap:8px 18px}aside li{margin:0}main{padding:36px}}
@media(max-width:540px){body{font-size:16px}.layout{padding:0 12px 30px}main{padding:26px 20px}h1{font-size:34px}h2{font-size:24px}.mast{padding:18px 24px}pre{font-size:12px;padding:14px}}
@media print{body{background:#fff;font-size:10.5pt}.mast,aside{display:none}.layout{display:block;margin:0;padding:0;max-width:none}main{border:0;box-shadow:none;padding:0}h1{font-size:26pt}h2{font-size:17pt;break-after:avoid}h3{font-size:13pt;break-after:avoid}tr,pre{break-inside:avoid}pre{white-space:pre-wrap}a{color:#16576a}table{font-size:9pt}.table-scroll{overflow:visible}}
"""
html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Evidence-backed review of Claude's gfx906 llama.cpp benchmarking project and prioritized patch proposals.">
<title>Vega 20: review and patch priorities</title><style>{css}</style></head>
<body><div class="mast">Engineering research · gfx906 / Vega 7nm</div>
<div class="layout"><aside aria-label="Report navigation"><h2>In this review</h2>{md.toc}<p><a href="README.md">Patch handoff and verification</a></p></aside>
<main>{body}<div class="footer">Prepared for Claude and the project owner. Research date: 8 September 2026.</div></main></div></body></html>'''
(root / "vega20-review.html").write_text(html)
print(root / "vega20-review.html")
