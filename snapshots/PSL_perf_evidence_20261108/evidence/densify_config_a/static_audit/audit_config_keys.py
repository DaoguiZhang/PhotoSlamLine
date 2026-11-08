#!/usr/bin/env python3
"""Static audit: settings_file[...] keys read by gaussian_mapper_line.cpp vs YAML keys."""
import re, csv, pathlib, sys, collections

root = pathlib.Path('/workspace/code/PSL-perf')
src = (root/'src/gaussian_mapper_line.cpp').read_text()
keys = sorted(set(re.findall(r'settings_file\["([^"]+)"\]', src)))
cfg_paths = [root/'cfg/gaussian_mapper/RGB-D/Replica/room1.yaml']
cfg_paths += sorted((root/'cfg/gaussian_mapper').rglob('*.yaml'))
seen = set(); cfgs = []
for p in cfg_paths:
    if p in seen: continue
    seen.add(p); cfgs.append(p)

def yaml_keys(path):
    out = {}
    for line in path.read_text(errors='replace').splitlines():
        if not line or line[0] in '#%' or ':' not in line: continue
        k, v = line.split(':', 1)
        k = k.strip()
        if k and ' ' not in k: out[k] = v.split('#',1)[0].strip()
    return out

rows=[]; missing_by_key=collections.Counter()
for p in cfgs:
    yk=yaml_keys(p)
    for k in keys:
        ok = k in yk
        if not ok: missing_by_key[k]+=1
        rows.append((str(p.relative_to(root)),k,'present' if ok else 'missing',yk.get(k,'')))

out=root/'evidence/densify_config_a/static_audit/config_key_presence.csv'
with out.open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['config','key','status','yaml_value']); w.writerows(rows)

summary=root/'evidence/densify_config_a/static_audit/key_summary.csv'
with summary.open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['key','configs_missing','configs_total','trailing_underscore','canonical_sibling'])
    for k in keys:
        w.writerow([k,missing_by_key[k],len(cfgs),int(k.endswith('_')),
                    k[:-1] if k.endswith('_') else ''])

print('configs',len(cfgs),'keys',len(keys))
print('keys missing from every config:')
for k in keys:
    if missing_by_key[k]==len(cfgs): print(' ',k)
print('keys with trailing underscore:')
for k in keys:
    if k.endswith('_'): print(' ',k,'sibling',k[:-1], 'missing_in',missing_by_key[k],'/',len(cfgs))
print('room1 missing count',sum(1 for r in rows if r[0].endswith('Replica/room1.yaml') and r[2]=='missing'))
