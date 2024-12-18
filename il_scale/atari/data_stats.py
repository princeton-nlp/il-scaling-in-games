import os
import time

import scipy.stats as stats
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

gameids = os.listdir("./datasets/BattleZone")
meta_schema = pa.schema([("score", pa.int32()), ("steps", pa.uint32())])

start = time.time()
dataset = pq.ParquetDataset(
    [os.path.join("./datasets/BattleZone", gid, "metadata.parquet") for gid in gameids],
    schema=meta_schema,
    use_legacy_dataset=False,
)
print(f"Loading took {time.time() - start} seconds")

start = time.time()
table = dataset.read(columns=["steps", "score"])
print(f"Table reading took {time.time() - start} seconds")

print(f"Total samples in dataset: {np.sum(table['steps'].to_numpy()):,}")
print(f"Mean samples in dataset: {np.mean(table['steps'].to_numpy()):,}")
print("Standard Error:", stats.sem(table["steps"].to_numpy()))
print("Mean score in dataset:", np.mean(table["score"].to_numpy()))
print("Standard Error:", stats.sem(table["score"].to_numpy()))
