# Standard library imports
import concurrent

# Third party imports
import torch
import numpy as np
import nle.dataset as nld
from nle.nethack.nethack import TERMINAL_SHAPE

# Local application imports
from il_scale.nethack.utils.setup import create_env
import il_scale.nethack.utils.constants as CONSTANTS

class TtyrecDataLoader:
    """
    Load a batch of ttyrec data into a torch tensor.
    """

    def __init__(
        self, 
        env_name: str, 
        device: torch.device, 
        dataset_name: str = 'nld-aa', 
        use_role: bool = False, 
        obs_frame_stack: int = 1, 
        use_inventory: bool = False,
        **dataset_kwargs
    ):
        # Create dataset
        self.dataset_name = dataset_name
        self.dataset = nld.TtyrecDataset(dataset_name, use_inventory=use_inventory, **dataset_kwargs)
        self.threadpool = dataset_kwargs["threadpool"] if "threadpool" in dataset_kwargs else None
        self.device = device
        self.use_role = use_role
        self.use_inventory = use_inventory
        self.gameids = self.dataset._gameids
        self.obs_frame_stack = obs_frame_stack

        # Create environment
        self.env = create_env(env_name, save_ttyrec_every=0)
        self.num_actions = len(self.env.actions)

        # Convert ASCII keypresses to action spaces indices
        embed_actions = torch.zeros((256, 1))
        for i, a in enumerate(self.env.actions):
            embed_actions[a.value][0] = i
        self.embed_actions = torch.nn.Embedding.from_pretrained(embed_actions)

        self.ttyrec_batch_size = dataset_kwargs['batch_size']
        self.unroll_length = dataset_kwargs['seq_length']
        self.prev_action_shape = (self.ttyrec_batch_size, self.unroll_length)

    def __iter__(self):
        """
        Returns a batch of ttyrec data.
        """
        return self.process_ttyrec_data_nld_aa()

    def process_ttyrec_data_nld_aa(self):
        def _iter():
            mb_tensors = {
                "prev_action": torch.zeros(self.prev_action_shape, dtype=torch.uint8)
            }

            prev_action = torch.zeros(
                (self.ttyrec_batch_size, 1), dtype=torch.uint8
            ).to(self.device)

            frame_stack_chars = torch.zeros((self.ttyrec_batch_size, self.obs_frame_stack - 1, TERMINAL_SHAPE[0], TERMINAL_SHAPE[1]))
            frame_stack_colors = frame_stack_chars.clone()

            for i, batch in enumerate(self.dataset):

                if i == 0:
                    # create torch tensors from first minibatch
                    # if self.use_screen:
                    #     screen_image = mb_tensors["screen_image"].numpy()
                    for k, array in batch.items():
                        mb_tensors[k] = torch.from_numpy(array)
                            
                    if self.device != torch.device('cpu'):
                        [v.pin_memory() for v in mb_tensors.values()]

                    if self.obs_frame_stack == 1:
                        mb_tensors['tty_chars'].unsqueeze_(2)
                        mb_tensors['tty_colors'].unsqueeze_(2)

                # Populate screen image
                cursor_uint8 = batch["tty_cursor"].astype(np.uint8)

                # Convert actions
                actions = mb_tensors["keypresses"].long()
                actions_converted = self.embed_actions(
                    actions).squeeze(-1).long().to(self.device)

                if self.use_role:
                    roles = torch.zeros_like(mb_tensors['gameids'])
                    for i in range(mb_tensors['gameids'].shape[0]):
                        for j in range(mb_tensors['gameids'].shape[1]):
                            gid = mb_tensors['gameids'][i][j].item()
                            if gid > 0:
                                meta = self.dataset.get_meta(gid)
                                roles[i][j] = CONSTANTS.ROLE_MAP[meta['role']]
                            else:
                                roles[i][j] = roles[i][0] # just role at beginning of time
                    mb_tensors['roles'] = roles

                final_mb = {
                    "tty_chars": mb_tensors["tty_chars"],
                    "tty_colors": mb_tensors["tty_colors"],
                    "tty_cursor": torch.from_numpy(cursor_uint8),
                    "scores": mb_tensors["scores"],
                    "done": mb_tensors["done"].bool(),
                    "labels": actions_converted,
                    "prev_action": torch.cat(
                        [prev_action, actions_converted[:, :-1]], dim=1
                    ),
                    "gameids": mb_tensors['gameids']
                }

                if self.obs_frame_stack > 1:
                    extended_chars = torch.cat([frame_stack_chars, mb_tensors["tty_chars"]], dim=1)
                    extended_colors = torch.cat([frame_stack_colors, mb_tensors["tty_colors"]], dim=1)

                    all_chars = []
                    all_colors = []
                    for t in range(self.unroll_length):
                        t_chars = extended_chars[:, t:t + self.obs_frame_stack]
                        t_colors = extended_colors[:, t:t + self.obs_frame_stack]
                        all_chars.append(t_chars)
                        all_colors.append(t_colors)
                    all_chars = torch.stack(all_chars, dim=1)
                    all_colors = torch.stack(all_colors, dim=1)
                    final_mb["tty_chars"] = all_chars
                    final_mb["tty_colors"] = all_colors

                if self.use_role:
                    final_mb["roles"] = mb_tensors['roles']

                if "blstats" in mb_tensors:
                    final_mb["blstats"] = mb_tensors["blstats"]

                if "inv_glyphs" in mb_tensors:
                    final_mb["inv_glyphs"] = mb_tensors["inv_glyphs"]

                prev_action = actions_converted[:, -1:]

                if self.obs_frame_stack > 1:
                    frame_stack_chars = mb_tensors["tty_chars"][:, -(self.obs_frame_stack - 1):].clone()
                    frame_stack_colors = mb_tensors["tty_colors"][:, -(self.obs_frame_stack - 1):].clone()
                    
                # Dataset is B x T, but model expects T x B
                yield {
                    k: t.transpose(0, 1).to(self.device)
                    for k, t in final_mb.items()
                }

        return _iter()
