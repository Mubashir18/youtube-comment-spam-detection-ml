import os
import pandas as pd

# Verify submission structure (this script is intended to be run from the submission `project/` folder)
required_files = {
    'Report': ['report.md', 'report.pdf'],
    'Scripts': ['download_data.py', 'run_all.py', 'generate_pdf_reportlab.py', 'verify_project.py'],
    'Code': ['src/preprocessing.py', 'src/models.py', 'src/evaluation.py', 'src/__init__.py'],
    'Data': ['data/youtube_spam.csv'],
    'Tests': ['test_reproducibility.py'],
    'Artifacts': [
        'artifacts/metrics-table.csv',
        'artifacts/error-examples.txt',
        'artifacts/plots/03_pr_curve_threshold_analysis.png',
        'artifacts/stress-tests/09_stress_test_1_character_noise.csv',
        'artifacts/stress-tests/10_stress_test_2_vocabulary_drift.csv',
        'artifacts/stress-tests/11_stress_test_3_adversarial_obfuscation.csv'
    ]
}

print("="*80)
print("PROJECT VERIFICATION REPORT")
print("="*80)

all_present = True
for category, files in required_files.items():
    print(f"\n{category}:")
    for file in files:
        exists = os.path.exists(file)
        status = "OK" if exists else "MISSING"
        print(f"  {status} {file}")
        all_present = all_present and exists

# Read and display key metrics
print("\n" + "="*80)
print("KEY METRICS")
print("="*80)

try:
    metrics_df = pd.read_csv('artifacts/metrics-table.csv')
    print("\nModel Comparison:")
    print(metrics_df.to_string(index=False))
    
    print("\n\nFinal Model Performance (Improved with SMOTE):")
    final_rows = metrics_df[metrics_df['model'].str.contains('Improved', regex=False)]
    if not final_rows.empty:
        final = final_rows.iloc[0]
        print(f"  Accuracy:         {final['accuracy']:.2%}")
        print(f"  Precision:        {final['precision']:.2%}")
        print(f"  Recall:           {final['recall']:.2%}")
        print(f"  F1 Score:         {final['f1']:.4f}")
        print(f"  Balanced Accuracy: {final['balanced_accuracy']:.2%}")
    else:
        print("Final model metrics not found in artifacts/metrics-table.csv")
    
except Exception as e:
    print(f"Error reading metrics: {e}")

# Stress tests summary
print("\n" + "="*80)
print("STRESS TEST SUMMARY")
print("="*80)

for test_file in ['artifacts/stress-tests/09_stress_test_1_character_noise.csv', 
                   'artifacts/stress-tests/10_stress_test_2_vocabulary_drift.csv',
                   'artifacts/stress-tests/11_stress_test_3_adversarial_obfuscation.csv']:
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        test_name = test_file.split('/')[-1].replace('.csv', '')
        print(f"\n{test_name}:")
        print(df.to_string(index=False))

print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)
if all_present:
    print("All required files present")
else:
    print("WARNING: Some files missing")

print("\nProject is READY FOR SUBMISSION")
print("="*80)
