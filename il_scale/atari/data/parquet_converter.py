import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

from il_scale.data.atari_schemas import OBS_SCHEMA


class ParquetConverter:
    def __init__(self):
        self.gameid = 0
        self.cursor = 0
        self.part = 0

    def load_parquet(self, path: str, gameid: int, part: int = None):
        self.gameid = gameid

        # Load stuff
        self.table = pq.read_table(
            path, columns=["states", "actions"], memory_map=True, schema=OBS_SCHEMA
        )
        num_rows = self.table.num_rows

        self.states = pa.compute.list_flatten(self.table["states"]).to_numpy()
        self.states = self.states.reshape(num_rows, 84, 84, 4)

        self.actions = self.table["actions"].to_numpy()

        # Reset cursor
        self.cursor = 0

    def convert(self, states, actions):
        input_len = states.shape[0]
        end_cursor = self.states.shape[0]
        to_read = min(input_len, end_cursor - self.cursor)

        np.copyto(states[:to_read], self.states[self.cursor : self.cursor + to_read])
        np.copyto(actions[:to_read], self.actions[self.cursor : self.cursor + to_read])

        self.cursor += to_read

        return input_len - to_read
