import pyarrow.parquet as pq

pf = pq.ParquetFile("/home/ap794/wspersonal/icu_digital_twins_mamba/data/patient_sequences/patient_sequences_2048_labeled.parquet")
print(pf.metadata.num_rows)