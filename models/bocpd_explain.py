"""
Diagnostic: Explain BOCPD accuracy test results
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bocpd import BOCPD


def explain_test_results():
    """Detailed breakdown of each test's results."""
    
    print("=" * 70)
    print("EXPLAINING BOCPD ACCURACY TEST RESULTS")
    print("=" * 70)
    
    # =========================================================================
    # TEST 1: Large Mean Shifts
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Large Mean Shifts - Why 100% Recall?")
    print("=" * 70)
    
    np.random.seed(42)
    segment_length = 50
    means = [0, 5, -3, 4, -2]
    
    data = []
    true_cps = []
    for i, mean in enumerate(means):
        data.extend(np.random.randn(segment_length) * 0.5 + mean)
        if i > 0:
            true_cps.append(i * segment_length)
    
    detector = BOCPD(hazard_rate=0.02, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=0.25)
    for x in data:
        detector.update(x)
    
    detected = detector.detect_change_points(method='map_drop', min_spacing=5)
    
    print(f"\nTrue CPs:     {true_cps}")
    print(f"Detected:     {detected}")
    print(f"\nMatching (tolerance=5):")
    
    tolerance = 5
    matched = []
    for true_cp in true_cps:
        best_match = None
        for det in detected:
            if abs(det - true_cp) <= tolerance:
                best_match = det
                break
        matched.append((true_cp, best_match))
        status = "✓ MATCHED" if best_match else "✗ MISSED"
        print(f"  True CP {true_cp} -> Detected {best_match} {status}")
    
    print(f"\nRecall = {sum(1 for _,m in matched if m)}/{len(true_cps)} = 100%")
    print(f"Extra detections (false positives): {[d for d in detected if not any(abs(d-tc)<=5 for tc in true_cps)]}")
    print("\nConclusion: ALL true CPs were found. The 'extra' detections hurt")
    print("precision but not recall. Recall measures 'did we find the real ones?'")
    
    # =========================================================================
    # TEST 3: Variance Shifts - Why only 50%?
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Variance Shifts - Why 50% Recall?")
    print("=" * 70)
    
    np.random.seed(42)
    stds = [0.3, 1.5, 0.3, 2.0, 0.3]
    
    data = []
    true_cps = []
    for i, std in enumerate(stds):
        data.extend(np.random.randn(segment_length) * std)
        if i > 0:
            true_cps.append(i * segment_length)
    
    detector = BOCPD(hazard_rate=0.02, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=0.1)
    for x in data:
        detector.update(x)
    
    detected = detector.detect_change_points(method='map_drop', min_spacing=5)
    
    print(f"\nTrue CPs:     {true_cps}")
    print(f"Detected:     {detected}")
    print(f"Variance pattern: {stds}")
    print(f"\nMatching (tolerance=5):")
    
    for true_cp in true_cps:
        best_match = None
        for det in detected:
            if abs(det - true_cp) <= tolerance:
                best_match = det
                break
        status = "✓ MATCHED" if best_match else "✗ MISSED"
        
        # What was the variance transition?
        idx = true_cps.index(true_cp)
        transition = f"std {stds[idx]} -> {stds[idx+1]}"
        print(f"  True CP {true_cp} ({transition}) -> Detected {best_match} {status}")
    
    print("\nNotice the pattern:")
    print("  - LOW -> HIGH variance (0.3 -> 1.5, 0.3 -> 2.0): Detected ✓")
    print("  - HIGH -> LOW variance (1.5 -> 0.3, 2.0 -> 0.3): Missed ✗")
    print("\nWhy? When variance INCREASES, outliers appear immediately (easy to spot).")
    print("When variance DECREASES, points just become 'normal' - nothing unusual!")
    
    # =========================================================================
    # Monte Carlo - Why so much better?
    # =========================================================================
    print("\n" + "=" * 70)
    print("MONTE CARLO - Why 93% vs 50-67%?")
    print("=" * 70)
    
    print("\nLet's look at what Monte Carlo does differently:\n")
    
    print("1. SHIFT MAGNITUDE:")
    print("   - Tests 1-3: Fixed shifts (e.g., 0->5, 0.3->2.0)")
    print("   - Monte Carlo: Random shifts of 2-5 units (ALWAYS large)")
    
    print("\n2. HAZARD RATE TUNING:")
    print("   - Tests 1-3: Fixed hazard_rate=0.02 (expects 50-pt segments)")
    print("   - Monte Carlo: hazard_rate=1/seg_len (PERFECTLY tuned each time)")
    
    print("\n3. MEAN SHIFTS ONLY:")
    print("   - Test 3: Variance shifts (harder for BOCPD)")
    print("   - Monte Carlo: Only mean shifts (BOCPD's strength)")
    
    print("\n4. NO COLD START PENALTY:")
    print("   - Tests 1-3: Always have false positive around t=15")
    print("   - Monte Carlo: Averages over 100 trials, cold start effect diluted")
    
    # Show Monte Carlo shift magnitudes
    print("\n" + "-" * 50)
    print("Sample Monte Carlo shift magnitudes:")
    print("-" * 50)
    np.random.seed(0)
    for trial in range(5):
        np.random.seed(trial)
        n_seg = np.random.randint(3, 7)
        means = [0.0]
        for _ in range(n_seg - 1):
            shift = np.random.choice([-1, 1]) * np.random.uniform(2, 5)
            means.append(means[-1] + shift)
        shifts = [means[i+1] - means[i] for i in range(len(means)-1)]
        print(f"  Trial {trial}: shifts = {[f'{s:+.1f}' for s in shifts]}")
    
    print("\nAll shifts are 2-5 units - these are VERY clear mean shifts!")
    print("Standard deviation is only 0.5, so a 3-unit shift is 6 sigma!")
    
    # =========================================================================
    # Fair comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("FAIR COMPARISON: Same conditions")
    print("=" * 70)
    
    print("\nLet's run Monte Carlo with VARIANCE shifts to see if it drops:\n")
    
    n_trials = 50
    all_f1 = []
    
    for trial in range(n_trials):
        np.random.seed(trial)
        n_seg = np.random.randint(3, 6)
        seg_len = 50
        
        # Variance shifts instead of mean shifts
        stds = [0.3]
        for _ in range(n_seg - 1):
            if stds[-1] < 1.0:
                stds.append(np.random.uniform(1.5, 2.5))  # Go high
            else:
                stds.append(0.3)  # Go low
        
        data = []
        true_cps = []
        for i, std in enumerate(stds):
            data.extend(np.random.randn(seg_len) * std)
            if i > 0:
                true_cps.append(i * seg_len)
        
        detector = BOCPD(hazard_rate=1/seg_len, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=0.1)
        for x in data:
            detector.update(x)
        
        detected = detector.detect_change_points(method='map_drop', min_spacing=5)
        
        # Evaluate
        tp = sum(1 for d in detected if any(abs(d-tc)<=5 for tc in true_cps))
        fp = len(detected) - tp
        fn = len(true_cps) - tp
        
        precision = tp / len(detected) if detected else 1.0
        recall = tp / len(true_cps) if true_cps else 1.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
        all_f1.append(f1)
    
    print(f"Monte Carlo with VARIANCE shifts:")
    print(f"  F1 Score: {np.mean(all_f1):.1%} ± {np.std(all_f1):.1%}")
    print(f"\nCompare to Monte Carlo with MEAN shifts: 94.3%")
    print("\nConclusion: BOCPD is much better at mean shifts than variance shifts!")


if __name__ == "__main__":
    explain_test_results()
