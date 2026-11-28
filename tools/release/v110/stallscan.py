# stallscan.py DIR... -- every llama-server log under DIR: the eval rates of its 256-token requests; a request is
# "slow" when it is more than 8 % under the median of its own log. Prints one line per log that has a slow request and
# totals per top-level directory.
import re, sys, os, statistics, time
pat = re.compile(r'\beval time =\s+([\d.]+) ms /\s+(\d+) tokens .*?([\d.]+) tokens per second')
tot = {}
for top in sys.argv[1:]:
    for root, _, files in os.walk(top):
        for fn in files:
            if not fn.endswith('.log'): continue
            p = os.path.join(root, fn)
            try:
                if os.path.getsize(p) > 200e6: continue
                rates = []
                for line in open(p, errors='replace'):
                    if 'prompt eval time' in line: continue
                    m = pat.search(line)
                    if m and int(m.group(2)) == 256: rates.append(float(m.group(3)))
            except OSError:
                continue
            if len(rates) < 2: continue
            med = statistics.median(rates)
            slow = [r for r in rates if r < 0.92 * med]
            key = top
            t = tot.setdefault(key, [0, 0, 0])
            t[0] += len(rates); t[1] += len(slow); t[2] += 1
            if slow:
                print(f"{time.strftime('%m-%d %H:%M', time.gmtime(os.path.getmtime(p)))} {p}: n {len(rates)} median {med:.1f} slow {' '.join('%.1f' % r for r in slow)}")
for k, t in tot.items():
    print(f"TOTAL {k}: {t[2]} logs, {t[0]} requests, {t[1]} slow")
