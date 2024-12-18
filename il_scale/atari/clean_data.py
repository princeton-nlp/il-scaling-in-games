import os

folders = os.listdir("datasets/Phoenix")
missing_folders = []
i = 0
for folder in folders:
    if not os.path.exists(
        f"datasets/Phoenix/{folder}/rollout.parquet"
    ) or not os.path.exists(f"datasets/Phoenix/{folder}/metadata.parquet"):
        missing_folders.append(folder)

    i += 1

    if i % 1000 == 0:
        print(f"Finished {i} folders")

print(missing_folders)
