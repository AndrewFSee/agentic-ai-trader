#!/usr/bin/env python3
import json

data = json.load(open('out_of_sample_results.json'))
print('\n✅ SUCCESSFUL STOCKS (15 out of 60):')
print('='*70)
successful = [r for r in data['results'] if r['success']]
for r in sorted(successful, key=lambda x: x['outperformance'], reverse=True):
    print(f"{r['symbol']:6} ({r['category']:20}) +{r['outperformance']:6.2f}% outperformance")
    print(f"       Strategy: {r['strategy_return']:6.2f}%  |  Buy-Hold: {r['buyhold_return']:6.2f}%")
print('\n')
print('='*70)
print('SUMMARY BY CATEGORY:')
print('='*70)
categories = {}
for r in data['results']:
    cat = r['category']
    if cat not in categories:
        categories[cat] = {'total': 0, 'success': 0, 'outperformance': []}
    categories[cat]['total'] += 1
    if r['success']:
        categories[cat]['success'] += 1
    categories[cat]['outperformance'].append(r['outperformance'])

for cat in sorted(categories.keys()):
    c = categories[cat]
    avg_out = sum(c['outperformance']) / len(c['outperformance'])
    success_rate = c['success']/c['total']*100
    print(f"{cat:25} {c['success']}/{c['total']} ({success_rate:3.0f}%)  Avg: {avg_out:+7.2f}%")
