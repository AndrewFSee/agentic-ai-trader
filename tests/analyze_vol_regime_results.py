#!/usr/bin/env python3
import json

data = json.load(open('volatility_regime_results.json'))

print('\n✅ SUCCESSFUL TESTS (7 out of 24):')
print('='*70)
successful = [r for r in data['results'] if r.get('success')]
for r in sorted(successful, key=lambda x: x['outperformance'], reverse=True):
    print(f"{r['symbol']:6} | {r['vol_regime']:6} Vol | {r['feature_set']:15} | +{r['outperformance']:5.2f}%")

print('\n\nRESULTS BY VOLATILITY REGIME:')
print('='*70)
regimes = {}
for r in data['results']:
    if 'error' in r:
        continue
    regime = r['vol_regime']
    if regime not in regimes:
        regimes[regime] = {'total': 0, 'success': 0, 'outperf': []}
    regimes[regime]['total'] += 1
    if r.get('success'):
        regimes[regime]['success'] += 1
    regimes[regime]['outperf'].append(r['outperformance'])

for regime in sorted(regimes.keys()):
    stats = regimes[regime]
    rate = stats['success'] / stats['total'] * 100
    avg = sum(stats['outperf']) / len(stats['outperf'])
    print(f"{regime.upper():8} Volatility: {stats['success']}/{stats['total']} ({rate:3.0f}%)  Avg: {avg:+6.2f}%")
