# wdscan.py FILE... -- per watchdog log: samples, samples with a busy die (>= 30 %) at <= 1000 MHz, the longest run of
# such samples in a row (24 = the latch criterion), the lowest clock of a busy die, the highest DC power.
import re, sys
pat = re.compile(r'card\d+:(\d+)MHz/(\d+)%')
for f in sys.argv[1:]:
    n = low = run = best = 0; mn = 99999; dc = 0
    for line in open(f, errors='replace'):
        c = pat.findall(line)
        if not c: continue
        n += 1
        m = re.search(r'DC:(\d+)W', line)
        if m: dc = max(dc, int(m.group(1)))
        busy = [int(a) for a, b in c if int(b) >= 30]
        if busy: mn = min(mn, min(busy))
        if any(x <= 1000 for x in busy):
            low += 1; run += 1; best = max(best, run)
        else:
            run = 0
    print(f"{f}: samples {n}, busy-die-at-1000 {low}, longest run {best}, lowest busy clock {mn if mn < 99999 else '-'} MHz, max DC {dc} W")
