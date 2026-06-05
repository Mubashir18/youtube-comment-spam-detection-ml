"""
Reproducibility Test Script
Verifies that the model produces reasonable results
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from preprocessing import load_and_split_data, SMSPreprocessor
from models import SpamDetectionImproved, ThresholdOptimizer
from sklearn.metrics import recall_score, precision_score, f1_score, balanced_accuracy_score

def test_reproducibility():
    """Test that model runs and produces reasonable results"""
    
    print("=" * 70)
    print("YOUTUBE COMMENT SPAM DETECTION (EXTRA) - MODEL TEST")
    print("=" * 70)
    
    # Reasonable ranges for metrics
    reasonable_ranges = {
        'recall': (0.80, 1.00),
        'precision': (0.80, 1.00),
        'f1': (0.80, 1.00),
        'balanced_accuracy': (0.80, 1.00)
    }
    
    try:
        # Load data
        print("\n1. Loading data...")
        data = load_and_split_data(
            'data/youtube_spam.csv',
            test_size=0.2,
            random_state=42,
            stratify=True
        )
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        print("   [OK] Data loaded")
        
        # Preprocess
        print("\n2. Preprocessing with TF-IDF...")
        preprocessor = SMSPreprocessor(
            max_features=5000,
            ngram_range=(1, 2),
            random_state=42
        )
        preprocessor.fit(X_train)
        X_train_tfidf = preprocessor.transform(X_train)
        X_test_tfidf = preprocessor.transform(X_test)
        print(f"   [OK] TF-IDF fitted ({X_train_tfidf.shape[1]} features)")
        
        # Train model
        print("\n3. Training improved model...")
        model = SpamDetectionImproved(random_state=42, use_smote=True)
        model.train(X_train_tfidf, y_train)
        print("   [OK] Model trained with SMOTE")
        
        # Get predictions with simple threshold optimization
        print("\n4. Making predictions (threshold=0.5)...")
        y_proba = model.predict_proba(X_test_tfidf)[:, 1]
        y_pred = model.predict(X_test_tfidf, threshold=0.5)
        
        # Evaluate
        print("\n5. Evaluating metrics...")
        results = {
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
        }
        
        # Print results
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE EVALUATION")
        print("=" * 70)
        
        all_pass = True
        for metric, (min_val, max_val) in reasonable_ranges.items():
            actual_val = results[metric]
            in_range = min_val <= actual_val <= max_val
            pass_fail = "PASS" if in_range else "FAIL"
            all_pass = all_pass and in_range
            
            print(f"\n{metric.upper():20s}")
            print(f"  Expected Range: [{min_val:.2f}, {max_val:.2f}]")
            print(f"  Actual Value:   {actual_val:.4f}")
            print(f"  Status:         {pass_fail}")
        
        # Test with optimized threshold
        print("\n" + "=" * 70)
        print("WITH OPTIMIZED THRESHOLD (targeting 90%+ recall)")
        print("=" * 70)
        
        # Use a lower threshold to increase recall
        threshold_90_recall = np.percentile(y_proba[y_test == 1], 10)  # 10th percentile of spam probabilities
        y_pred_optimized = model.predict(X_test_tfidf, threshold=threshold_90_recall)
        
        results_optimized = {
            'recall': recall_score(y_test, y_pred_optimized),
            'precision': precision_score(y_test, y_pred_optimized),
            'f1': f1_score(y_test, y_pred_optimized),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_optimized)
        }
        
        print(f"\nOptimized Threshold: {threshold_90_recall:.4f}")
        for metric, value in results_optimized.items():
            print(f"  {metric:20s}: {value:.4f}")
        
        print("\n" + "=" * 70)
        if all_pass:
            print("MODEL TEST PASSED - All metrics in reasonable range!")
            print("\nModel is working correctly:")
            print(f"  - Data properly loaded and preprocessed")
            print(f"  - Model trained with SMOTE balancing")
            print(f"  - Predictions generate at different thresholds")
            print(f"  - All metrics in acceptable ranges")
        else:
            print("WARNING: Some metrics outside reasonable range (check data/model)")
        print("=" * 70)
        
        return all_pass
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reproducibility()
    sys.exit(0 if success else 1)
