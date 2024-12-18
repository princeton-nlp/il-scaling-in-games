import os
import random

import torch

game = "SpaceInvaders"

gameids = os.listdir(f"./datasets/{game}")
gameids = list(map(lambda x: int(x), gameids))
# shuffle gameids
random.shuffle(gameids)
train_gids = gameids[:-1]
dev_gids = gameids[-1:]

assert len(dev_gids) == 1
assert len(train_gids) + len(dev_gids) == len(gameids)

# save
torch.save(
    {"train_gids": train_gids, "dev_gids": dev_gids},
    f"data_objects/{game}_data_split.tar",
)
