import re, sys

log = open(sys.argv[1] if len(sys.argv) > 1 else 'abl_out/gold_r0/run.log',
           errors='ignore').read().splitlines()
recs, cur, i = [], None, 0
while i < len(log):
    m = re.search(r'\[AggSelfCheck\] P=(\d+)', log[i])
    if m:
        cur = {'P': int(m.group(1)), 'd': {}}
        recs.append(cur)
    if 'vs FP64 golden' in log[i] and cur is not None:
        for k in range(1, 13):
            mm = re.match(r'\s+([ABC]) (\S+)\s+max_abs=([\d.e+-]+) relL2=([\d.e+-]+)',
                          log[i + k])
            if mm:
                cur['d'][(mm.group(1), mm.group(2))] = (float(mm.group(3)),
                                                        float(mm.group(4)))
        i += 12
    i += 1

seen, cases = set(), []
for r in recs:
    if r['d'] and r['P'] not in seen:
        seen.add(r['P'])
        cases.append(r)
cases.sort(key=lambda x: x['P'])

hdr = (f"{'P':>8} | {'group':>8} | {'A max_abs':>10} {'C max_abs':>10} | "
       f"{'A relL2':>10} {'C relL2':>10} | {'C/A':>5}")
print(hdr)
print('-' * len(hdr))
worst = 0.0
for r in cases:
    for g in ('mean2D', 'conic2D', 'opacity', 'colors'):
        if ('A', g) not in r['d'] or ('C', g) not in r['d']:
            continue
        A, C = r['d'][('A', g)], r['d'][('C', g)]
        ratio = C[1] / A[1] if A[1] else float('nan')
        worst = max(worst, ratio)
        print(f"{r['P']:>8} | {g:>8} | {A[0]:10.3e} {C[0]:10.3e} | "
              f"{A[1]:10.3e} {C[1]:10.3e} | {ratio:5.2f}")
print(f"\ncases={len(cases)}  worst C/A relL2 ratio = {worst:.2f}")
print("A relL2 range: %.3e .. %.3e" % (min(v[1] for r in cases for k, v in r['d'].items() if k[0] == 'A'),
                                      max(v[1] for r in cases for k, v in r['d'].items() if k[0] == 'A')))
print("C relL2 range: %.3e .. %.3e" % (min(v[1] for r in cases for k, v in r['d'].items() if k[0] == 'C'),
                                      max(v[1] for r in cases for k, v in r['d'].items() if k[0] == 'C')))
