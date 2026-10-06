import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
df = pd.read_csv(os.path.join(PROJECT_ROOT, "datasets", "original_datasets", "bank.csv"))

# Complain vs Exited
print("=== COMPLAIN vs Exited ===")
print(pd.crosstab(df['Complain'], df['Exited'], normalize='index') * 100)

print("\n=== SATISFACTION SCORE vs Exited ===")
print(pd.crosstab(df['Satisfaction Score'], df['Exited'], normalize='index') * 100)

# Raw counts
print("\n=== RAW COUNTS: COMPLAIN ===")
print(pd.crosstab(df['Complain'], df['Exited']))

print("\n=== CORRELATION WITH Exited ===")
print(df[['Complain', 'Satisfaction Score', 'Exited']].corr()['Exited'])
