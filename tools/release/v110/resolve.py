#!/usr/bin/env python3
"""resolve.py FILE N HOW -- replace the N-th (1-based, in the file as it is now) conflict hunk.
HOW: ours | theirs | both (ours then theirs) | file:PATH (the text of PATH) | show (print it, change nothing).
Prints the number of hunks left."""
import sys, re
path, n, how = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines = open(path, encoding='utf-8').read().split('\n')
hunks = []; i = 0
while i < len(lines):
    if lines[i].startswith('<<<<<<< '):
        s = i
        while not lines[i].startswith('======='): i += 1
        m = i
        while not lines[i].startswith('>>>>>>> '): i += 1
        hunks.append((s, m, i))
    i += 1
if not 1 <= n <= len(hunks): sys.exit(f'{path}: {len(hunks)} hunks, no hunk {n}')
s, m, e = hunks[n - 1]
ours, theirs = lines[s + 1:m], lines[m + 1:e]
if how == 'show':
    print(f'--- {path} hunk {n} of {len(hunks)}: lines {s + 1}-{e + 1}\n<<< OURS ({len(ours)} lines)')
    print('\n'.join(ours)); print(f'=== THEIRS ({len(theirs)} lines)'); print('\n'.join(theirs)); sys.exit(0)
if how == 'ours': new = ours
elif how == 'theirs': new = theirs
elif how == 'both': new = ours + theirs
elif how.startswith('file:'):
    new = open(how[5:], encoding='utf-8').read().split('\n')
    if new and new[-1] == '': new.pop()
else: sys.exit('HOW?')
lines[s:e + 1] = new
open(path, 'w', encoding='utf-8').write('\n'.join(lines))
print(f'{path}: hunk {n} -> {how} ({len(new)} lines); {len(hunks) - 1} left')
