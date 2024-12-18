import os
import sqlite3
import threading
from collections import defaultdict
from functools import partial

import numpy as np

from il_scale.data.parquet_converter import ParquetConverter
from il_scale.utils.atari_conf import OBS_SHAPE


def convert_frames(converter, states, actions, resets, gameids, load_fn):
    """Convert frames for a single batch entry.

    :param converter: A ttychars.Converter object with a loaded file.
    :param states: Array of states -  np.array(np.int16) [ SEQ x ROW x COL]
    :param actions: Array of actions at t in response to output at t
        - np.array(np.uint8) [ SEQ ]
    :param resets: Array of resets -  np.array(np.uint8) [ SEQ ]
    :param gameids: Array of the gameid of each frame - np.array(np.int32) [ SEQ ]
    :param load_fn: A callback that loads the next file into a converter:
        sig: load_fn(converter) -> bool is_success

    """
    resets[0] = 0
    while True:
        remaining = converter.convert(states, actions)
        end = np.shape(states)[0] - remaining

        resets[1:end] = 0
        gameids[:end] = converter.gameid
        if remaining == 0:
            return

        # There still space in the buffers; load a new parquet and carry on.
        states = states[-remaining:]
        actions = actions[-remaining:]
        resets = resets[-remaining:]
        gameids = gameids[-remaining:]
        if load_fn(converter):
            if converter.part == 0:
                resets[0] = 1
        else:
            states.fill(0)
            actions.fill(0)
            resets.fill(0)
            gameids.fill(0)
            return


def _parquet_generator(batch_size, seq_length, load_fn, map_fn):
    """A generator to fill minibatches with ttyrecs.

    :param load_fn: a function to load the next ttyrec into a converter.
       load_fn(ttyrecs.Converter conv) -> bool is_success
    :param map_fn: a function that maps a series of iterables through a fn.
       map_fn(fn, *iterables) -> <generator> (can use built-in map)

    """
    states = np.zeros(
        (batch_size, seq_length, OBS_SHAPE[0], OBS_SHAPE[1], OBS_SHAPE[2]),
        dtype=np.int16,
    )
    actions = np.zeros((batch_size, seq_length), dtype=np.uint8)
    resets = np.zeros((batch_size, seq_length), dtype=np.uint8)
    gameids = np.zeros((batch_size, seq_length), dtype=np.int32)

    key_vals = [
        ("states", states),
        ("actions", actions),
        ("done", resets),
        ("gameids", gameids),
    ]

    # Load initial gameids.
    converters = [ParquetConverter() for _ in range(batch_size)]
    # assert all(load_fn(c) for c in converters), "Not enough ttyrecs to fill a batch!"
    assert all(list(map_fn(load_fn, converters))), "Not enough states to fill a batch!"

    # Convert (at least one minibatch)
    _convert_frames = partial(convert_frames, load_fn=load_fn)
    gameids[0, -1] = 1  # basically creating a "do-while" loop by setting an indicator
    while np.any(
        gameids[:, -1] != 0
    ):  # loop until only padding is found, i.e. end of data
        list(map_fn(_convert_frames, converters, states, actions, resets, gameids))

        yield dict(key_vals)


class ParquetDataset:
    """Dataset object to allow iteration through parquet files."""

    def __init__(
        self,
        dataset_root: str,
        batch_size=128,
        seq_length=32,
        threadpool=None,
        gameids=None,
        shuffle=True,
        loop_forever=False,
    ):
        """
        :param dataset_root: Path to root dataset folder.
        :param batch_size: Number of parallel games to load.
        :param seq_length: Number of frames to load per game.
        :param gameids: Use a subselection of games (gameids) only.
        :param shuffle: Shuffle the order of gameids before iterating through them.
        :param loop_forever: If true, cycle through gameids forever,
            insted of padding empty batch dims with 0's.
        """
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.shuffle = shuffle
        self.loop_forever = loop_forever
        self.dataset_root = dataset_root

        if gameids is None:
            gameids = list(map(lambda x: int(x), os.listdir(self.dataset_root)))

        self._gameids = list(gameids)
        self._threadpool = threadpool
        self._map = partial(self._threadpool.map, timeout=12000) if threadpool else map

    def _make_load_fn(self, gameids):
        """Make a closure to load the next gameid from the db into the converter."""
        lock = threading.Lock()
        count = [0]

        def _load_fn(converter, move_game=False):
            """Take the next part of the current game if available, else new game.
            Return True if load successful, else False."""
            gameid = converter.gameid
            part = converter.part + 1

            files = self.get_paths(gameid)
            if gameid == 0 or part >= len(files) or move_game:
                with lock:
                    i = count[0]
                    count[0] += 1

                if (not self.loop_forever) and i >= len(gameids):
                    return False

                gameid = gameids[i % len(gameids)]
                files = self.get_paths(gameid)
                part = 0

            filename = files[part]
            converter.load_parquet(filename, gameid=gameid, part=part)
            return True

        return _load_fn

    def get_paths(self, gameid: int):
        if gameid == 0:
            return []
        folder = os.path.join(self.dataset_root, str(gameid))
        files = os.listdir(folder)
        rollout_files = [
            os.path.join(folder, f) for f in files if f.startswith("rollout")
        ]
        return sorted(rollout_files)

    def __iter__(self):
        gameids = list(self._gameids)
        if self.shuffle:
            np.random.shuffle(gameids)

        return _parquet_generator(
            self.batch_size, self.seq_length, self._make_load_fn(gameids), self._map
        )
