from concurrent.futures import ThreadPoolExecutor
import time
import os

import torch

from il_scale.data.parquet_dataset import ParquetDataset

B = 128
T = 32
tp = ThreadPoolExecutor(max_workers=30)
dataset = ParquetDataset(
    dataset_root=f"datasets/BattleZone",
    threadpool=tp,
    seq_length=T,
    batch_size=B,
    # gameids=torch.load('data_objects/Boxing_data_split.tar')['train_gids'],
    gameids=list(map(lambda x: int(x), os.listdir("datasets/BattleZone/"))),
)

freq = 50
start = time.time()
i = 1
for batch in dataset:
    if i % freq == 0:
        print(f"fps: {B * T * freq / (time.time() - start)}")
        start = time.time()

    i += 1
