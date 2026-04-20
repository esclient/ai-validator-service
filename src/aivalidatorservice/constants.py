import pandas as pd
for f in ["russian_dataset_2.tsv", "russian_distorted_toxicity.tsv"]:
    df = pd.read_csv(f"data/datasets/{f}", sep="\t", nrows=3)
    print(f"\n{f}:")
    print(df.columns.tolist())
    print(df.head(2))
