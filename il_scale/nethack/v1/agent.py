from typing import Union

from nle import nethack
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from il_scale.nethack.v1.config import Config
from il_scale.nethack.v1.net import NetHackNetTtyrec


class Agent:
    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger

        # Assign initial dummy rank and world_size
        self.rank = 0
        self.world_size = 1
        self.ddp = False

    def predict(self, batch, agent_state):
        return self.model(batch, agent_state)

    def initial_state(
        self, batch_size: int, device: Union[int, torch.device] = torch.device("cpu")
    ):
        model = self.model.module if self.ddp else self.model
        if model.use_lstm:
            agent_state = model.initial_state(batch_size=batch_size)
            agent_state = (agent_state[0].to(device), agent_state[1].to(device))
        else:
            agent_state = None

        return agent_state

    def move_to_ddp(
        self, rank: int, world_size: int, find_unused_parameters: bool = True
    ):
        self.rank = rank
        self.world_size = world_size
        self.ddp = True
        self.model = DDP(
            self.model, device_ids=[rank], find_unused_parameters=find_unused_parameters
        )

    def load(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model.to(device)

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def state_dict(self):
        return self.model.module.state_dict() if self.ddp else self.model.state_dict()

    def construct_model(self, load_config=None):
        config = self.config if not load_config else load_config

        self.model = NetHackNetTtyrec(
            len(nethack.ACTIONS),
            use_lstm=config.use_lstm,
            use_policy_and_value_heads=True,
            il_mode=True,
            h_dim=config.h_dim,
            msg_hdim=config.msg_hdim,
            scale_cnn_channels=config.scale_cnn_channels,
            num_lstm_layers=config.num_lstm_layers,
            num_fc_layers=config.num_fc_layers,
            num_screen_fc_layers=config.num_screen_fc_layers,
            use_prev_action=config.use_prev_action,
            color_edim=config.color_edim,
            char_edim=config.char_edim,
            use_crop=config.use_crop,
            use_charcnn_topline=config.use_charcnn_topline,
            use_parsed_blstats=config.use_parsed_blstats,
            use_condition_bits=config.use_condition_bits,
            use_resnet=config.use_resnet,
            use_crop_norm=config.use_crop_norm,
            blstats_version=config.blstats_version,
            use_bl_role=config.use_bl_role,
            use_inventory=config.use_inventory,
            lagged_actions=config.lagged_actions,
            obs_frame_stack=config.obs_frame_stack,
            num_res_blocks=config.num_res_blocks,
            num_res_layers=config.num_res_layers,
            use_cur_action=config.use_cur_action,
            screen_kernel_size=config.screen_kernel_size,
            no_max_pool=config.no_max_pool,
            screen_conv_blocks=config.screen_conv_blocks,
            blstats_hdim=config.blstats_hdim,
            fc_after_cnn_hdim=config.fc_after_cnn_hdim,
        )
