#!/usr/bin/env python3
"""Static audit: settings_file[...] keys read by gaussian_mapper_line.cpp vs YAML.

Read-style classification is deliberately conservative:
  guarded  : the same key appears on a line containing `.isNone()` (explicit default)
  strict   : the key is consumed by readDensifyFromIter() (canonical/legacy resolver)
  direct   : a bare operator int()/float()/... read
A missing direct-read key is the class that can silently become 0 via OpenCV.
"""
import re, csv, pathlib, collections

root = pathlib.Path('/workspace/code/PSL-perf')
src = pathlib.Path('/tmp/psl-densify-a/src/gaussian_mapper_line.cpp').read_text()
keys = set(re.findall(r'settings_file\["([^"]+)"\]', src))
# Densify from-iter is read by readDensifyFromIter(file, ...), not via
# settings_file[...], so add both resolver keys explicitly.
keys |= {'Optimization.densify_from_iter', 'Optimization.densify_from_iter_'}
keys = sorted(keys)

style = {}
for key in keys:
    if key in ('Optimization.densify_from_iter', 'Optimization.densify_from_iter_'):
        style[key] = 'strict_canonical_legacy_resolver'
        continue
    idx = src.find(f'settings_file["{key}"]')
    if idx < 0:
        style[key] = 'unknown'
        continue
    lo = src.rfind('\n', 0, idx)
    hi = src.find(';', idx)
    stmt = src[lo:hi if hi >= 0 else idx + 200]
    if '.isNone()' in stmt:
        style[key] = 'guarded_explicit_default'
    else:
        style[key] = 'direct_read'

cfg_paths = [root / 'cfg/gaussian_mapper/RGB-D/Replica/room1.yaml']
cfg_paths += sorted((root / 'cfg/gaussian_mapper').rglob('*.yaml'))
seen = set(); cfgs = []
for p in cfg_paths:
    rp = p.resolve()
    if rp in seen:
        continue
    seen.add(rp); cfgs.append(p)

def yaml_keys(path):
    out = {}
    for line in path.read_text(errors='replace').splitlines():
        if not line or line[0] in '#%' or ':' not in line:
            continue
        k, v = line.split(':', 1)
        k = k.strip()
        if k and ' ' not in k:
            out[k] = v.split('#', 1)[0].strip()
    return out

rows = []
missing_by_key = collections.Counter()
for p in cfgs:
    yk = yaml_keys(p)
    for k in keys:
        ok = k in yk
        if not ok:
            missing_by_key[k] += 1
        rows.append((str(p.relative_to(root)), k, 'present' if ok else 'missing',
                     style.get(k, 'unknown'), yk.get(k, '')))

out_dir = root / 'evidence/densify_config_a/static_audit'
with (out_dir / 'config_key_presence.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['config', 'key', 'status', 'read_style', 'yaml_value'])
    w.writerows(rows)

with (out_dir / 'key_summary.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['key', 'read_style', 'configs_missing', 'configs_total',
                'trailing_underscore', 'canonical_sibling'])
    for k in keys:
        w.writerow([k, style.get(k, 'unknown'), missing_by_key[k], len(cfgs),
                    int(k.endswith('_')), k[:-1] if k.endswith('_') else ''])

room1 = [r for r in rows if r[0].endswith('Replica/room1.yaml')]
room1_missing = [r for r in room1 if r[2] == 'missing']
with (out_dir / 'room1_missing_keys.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['config', 'key', 'read_style', 'yaml_value'])
    for r in room1_missing:
        w.writerow([r[0], r[1], r[3], r[4]])

with (out_dir / 'summary.md').open('w') as f:
    f.write('# Static config-key audit\n\n')
    f.write(f'- configs: {len(cfgs)}\n- keys read by `gaussian_mapper_line.cpp`: {len(keys)}\n')
    f.write(f'- room1 missing keys: {len(room1_missing)}\n\n')
    f.write('## Room1 missing keys\n\n| key | read style |\n|---|---|\n')
    for r in room1_missing:
        f.write(f'| `{r[1]}` | {r[3]} |\n')
    f.write('\n## Keys missing from every gaussian YAML\n\n')
    for k in keys:
        if missing_by_key[k] == len(cfgs):
            f.write(f'- `{k}` ({style.get(k)})\n')
    f.write('\n## Direct reads missing in any config\n\n')
    for k in keys:
        if style.get(k) == 'direct_read' and missing_by_key[k] > 0:
            f.write(f'- `{k}` missing in {missing_by_key[k]}/{len(cfgs)}\n')

print('configs', len(cfgs), 'keys', len(keys), 'room1_missing', len(room1_missing))
print('room1 missing:', ', '.join(r[1] for r in room1_missing))
print('direct read missing anywhere:')
for k in keys:
    if style.get(k) == 'direct_read' and missing_by_key[k] > 0:
        print(' ', k, missing_by_key[k], '/', len(cfgs))
print('missing everywhere:')
for k in keys:
    if missing_by_key[k] == len(cfgs):
        print(' ', k, style.get(k))
