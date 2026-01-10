import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("./datasets/DiseaseAndSymptoms.csv")

# 1. Clean whitespace from all symptom columns
cols = data.columns[1:]
for col in cols:
    data[col] = data[col].str.strip()

# 2. Get unique symptoms
all_symptoms = pd.unique(data[cols].values.ravel('K'))
all_symptoms = [s for s in all_symptoms if str(s) != 'nan']
print(f"Total unique symptoms: {len(all_symptoms)}")

# 3. Create a binary dataframe (tidy format)
binary_df = pd.DataFrame(columns=all_symptoms)
binary_df.insert(0, 'Disease', data['Disease'])

# Fill the binary dataframe
# This might be slow for large datasets, let's try a more vectorized approach if possible
# But for now, let's just see the unique symptoms and first few rows

print("Unique Symptoms Sample:", all_symptoms[:10])
print("Disease unique counts:")
print(data['Disease'].value_counts())
