import pandas as pd

df = pd.read_parquet("/hpc/home/aparedeslatorre1/icu_digital_twins_mamba/data/patient_sequences/patient_sequences_2048.parquet")
print(df.columns)
print(df.head(3))


import pickle
import pandas as pd

# Read the pickle file
with open('/hpc/home/aparedeslatorre1/icu_digital_twins_mamba/data/patient_id_dict/dataset_2048_multi_v2.pkl', 'rb') as f:
    patient_dict = pickle.load(f)

# Read the parquet file
patient_sequences = pd.read_parquet('/hpc/home/aparedeslatorre1/icu_digital_twins_mamba/data/patient_sequences/patient_sequences_2048.parquet')

# Now you can explore the data
print("Patient dictionary type:", type(patient_dict))
print("Patient dictionary keys (if it's a dict):", patient_dict.keys() if isinstance(patient_dict, dict) else "Not a dictionary")
print("\nPatient sequences shape:", patient_dict['pretrain'])


print("Patient sequences columns:", patient_sequences.columns.tolist())
print("\nFirst few rows of patient sequences:")
print(patient_sequences.head())