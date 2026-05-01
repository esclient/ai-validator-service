import pandas as pd

from custom_logger import get_logger

log = get_logger(__name__)

for f in ["russian_dataset_2.tsv", "russian_distorted_toxicity.tsv"]:
    df = pd.read_csv(f"data/datasets/{f}", sep="\t", nrows=3)
    log.info(f"Previewing dataset file: {f}")
    log.debug(f"Columns: {df.columns.tolist()}")
    log.debug(f"Head preview:\n{df.head(2)}")
