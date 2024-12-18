import pyarrow as pa

obs_schema_keys = [
    ("states", pa.list_(pa.int16())),
    ("actions", pa.uint8()),
]
OBS_SCHEMA = pa.schema(obs_schema_keys)
