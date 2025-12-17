#!/usr/bin/env python3
"""Quick test to debug MRVL issue"""

from test_hmm_features import HMMFeatureTester
import traceback

try:
    print('Testing MRVL...')
    tester = HMMFeatureTester('MRVL', 730)
    tester.prepare_data()
    print('Data prepared successfully')
    
    # Test with just one simple feature
    results = tester.grid_search([['macd_hist_norm']], regime_counts=[3])
    print('Test completed successfully')
    print(results[0])
    
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc()
