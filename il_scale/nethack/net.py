import logging
import math
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from nle import nethack  # noqa: E402
from nle.nethack.nethack import TERMINAL_SHAPE
from transformers import TopPLogitsWarper, TopKLogitsWarper

from il_scale.nethack.crop import Crop
from il_scale.nethack.utils import interleave

PAD_CHAR = 0
NUM_CHARS = 256

class NetHackNetTtyrec(nn.Module):
    """
    Adapted from https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022/blob/67139262966aa11555cf7aca15723375b36fbe42/experiment_code/hackrl/models/offline_chaotic_dwarf.py
    """
    def __init__(
        self, 
        num_actions: int, 
        h_dim: int = 512, 
        use_lstm: bool = False,
        msg_hdim: int = 64,
        use_prev_action: bool = False,
        use_policy_and_value_heads: bool = True,
        il_mode: bool = False,
        scale_cnn_channels: int = 1,
        num_lstm_layers: int = 1,
        num_fc_layers: int = 2,
        num_screen_fc_layers: int = 1,
        sampling_type: str = 'softmax',
        color_edim: int = 16,
        char_edim: int = 16,
        crop_dim: int = 9,
        crop_out_filters: int = 8,
        crop_num_layers: int = 5,
        crop_inter_filters: int = 16,
        crop_padding: int = 1,
        crop_kernel_size: int = 3,
        crop_stride: int = 1,
        use_crop: bool = False,
        use_charcnn_topline: bool = False,
        use_parsed_blstats: bool = False,
        use_condition_bits: bool = False,
        use_bl_role: bool = False,
        use_encumbrance: bool = False,
        temperature: float = 1,
        normalize_blstats: bool = False,
        use_resnet: bool = False,
        use_crop_norm: bool = False,
        blstats_version: str = "v2",
        use_inventory: bool = False,
        lagged_actions: int = 0,
        action_embedding_dim: int = 32,
        top_p: float = 0.9,
        top_k: int = 1,
        obs_frame_stack: int = 1,
        num_res_blocks: int = 2,
        num_res_layers: int = 2,
        use_cur_action: bool = False,
        screen_kernel_size: int = 3,
        no_max_pool: bool = False,
        screen_conv_blocks: int = 3,
        blstats_hdim: int = None,
        fc_after_cnn_hdim: int = None
    ):
        super(NetHackNetTtyrec, self).__init__()

        self.register_buffer("reward_sum", torch.zeros(()))
        self.register_buffer("reward_m2", torch.zeros(()))
        self.register_buffer("reward_count", torch.zeros(()).fill_(1e-8))

        self.num_actions = num_actions 
        self.use_prev_action = use_prev_action
        self.use_cur_action = use_cur_action
        self.use_lstm = use_lstm
        self.msg_hdim = msg_hdim
        self.h_dim = h_dim
        self.use_policy_and_value_heads = use_policy_and_value_heads
        self.il_mode = il_mode
        self.scale_cnn_channels = scale_cnn_channels
        self.num_lstm_layers = num_lstm_layers
        self.num_fc_layers = num_fc_layers
        self.num_screen_fc_layers = num_screen_fc_layers
        self.sampling_type = sampling_type
        self.color_edim = color_edim 
        self.char_edim = char_edim
        self.crop_dim = crop_dim
        self.crop_out_filters = crop_out_filters
        self.crop_num_layers = crop_num_layers
        self.crop_inter_filters = crop_inter_filters
        self.crop_padding = crop_padding
        self.crop_kernel_size = crop_kernel_size
        self.crop_stride = crop_stride
        self.use_crop = use_crop
        self.use_charcnn_topline = use_charcnn_topline
        self.use_parsed_blstats = use_parsed_blstats
        self.use_condition_bits = use_condition_bits
        self.use_bl_role = use_bl_role
        self.use_encumbrance = use_encumbrance
        self.temperature = temperature
        self.normalize_blstats = normalize_blstats
        self.use_resnet = use_resnet
        self.use_crop_norm = use_crop_norm
        self.blstats_version = blstats_version
        self.use_inventory = use_inventory
        self.lagged_actions = lagged_actions
        self.action_embedding_dim = action_embedding_dim
        self.top_k = top_k
        self.top_p = top_p
        self.obs_frame_stack = obs_frame_stack
        self.screen_shape = TERMINAL_SHAPE
        self.screen_kernel_size = screen_kernel_size
        self.no_max_pool = no_max_pool
        self.screen_conv_blocks = screen_conv_blocks
        self.blstats_hdim = blstats_hdim if blstats_hdim else h_dim
        self.fc_after_cnn_hdim = fc_after_cnn_hdim if fc_after_cnn_hdim else h_dim

        self.top_p_logits_warper = TopPLogitsWarper(self.top_p)
        self.top_k_logits_warper = TopKLogitsWarper(self.top_k)
    
        # NOTE: -3 because we cut the topline and bottom two lines
        if self.use_crop:
            self.crop = Crop(self.screen_shape[0] - 3, self.screen_shape[1], self.crop_dim, self.crop_dim)
            crop_in_channels = [self.char_edim + self.color_edim] + [self.crop_inter_filters] * (self.crop_num_layers - 1)
            crop_out_channels = [self.crop_inter_filters] * (self.crop_num_layers - 1) + [self.crop_out_filters]
            conv_extract_crop = []
            norm_extract_crop = []
            for i in range(self.crop_num_layers):
                conv_extract_crop.append(nn.Conv2d(
                    in_channels=crop_in_channels[i],
                    out_channels=crop_out_channels[i],
                    kernel_size=(self.crop_kernel_size, self.crop_kernel_size),
                    stride=self.crop_stride,
                    padding=self.crop_padding,
                ))
                norm_extract_crop.append(nn.BatchNorm2d(crop_out_channels[i]))

            if self.use_crop_norm:
                self.extract_crop_representation = nn.Sequential(
                    *interleave(conv_extract_crop, norm_extract_crop, [nn.ELU()] * len(conv_extract_crop))
                )
            else:
                self.extract_crop_representation = nn.Sequential(
                    *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract_crop))
                )
            self.crop_out_dim = self.crop_dim ** 2 * self.crop_out_filters
        else:
            self.crop_out_dim = 0


        self.topline_encoder = TopLineEncoder(msg_hdim=self.msg_hdim)
        self.bottomline_encoder = BottomLinesEncoder(h_dim=self.blstats_hdim//4)

        
        self.inventory_dim = 0

        self.screen_encoder = CharColorEncoderResnet(
            (self.screen_shape[0] - 3, self.screen_shape[1]), 
            h_dim=self.fc_after_cnn_hdim,
            num_fc_layers=self.num_screen_fc_layers, 
            scale_cnn_channels=self.scale_cnn_channels,
            color_edim=color_edim,
            char_edim=char_edim,
            obs_frame_stack=obs_frame_stack,
            num_res_blocks=num_res_blocks,
            num_res_layers=num_res_layers,
            kernel_size=screen_kernel_size,
            no_max_pool=self.no_max_pool,
            screen_conv_blocks=self.screen_conv_blocks
        )

        self.prev_actions_dim = self.num_actions if self.use_prev_action else 0
        self.cur_actions_dim = self.num_actions if self.use_cur_action else 0

        if self.lagged_actions:
            self.action_embed = nn.Embedding(num_actions + 2, action_embedding_dim)

        self.out_dim = sum(
            [
                self.topline_encoder.msg_hdim,
                self.bottomline_encoder.h_dim,
                self.screen_encoder.h_dim,
                self.prev_actions_dim,
                self.cur_actions_dim,
                self.crop_out_dim,
                self.inventory_dim,
                self.lagged_actions * self.action_embedding_dim
            ]
        )

        if self.use_lstm:
            self.core = nn.LSTM(self.h_dim, self.h_dim, num_layers=self.num_lstm_layers)
    

        if use_policy_and_value_heads:
            fc_layers = [nn.Linear(self.out_dim, self.h_dim), nn.ReLU()]
            for _ in range(self.num_fc_layers - 1):
                fc_layers.append(nn.Linear(self.h_dim, self.h_dim))
                fc_layers.append(nn.ReLU())
            self.fc = nn.Sequential(*fc_layers)

            self.policy = nn.Linear(self.h_dim, self.num_actions)
            self.baseline = nn.Linear(self.h_dim, 1)
        self.version = 0

    def initial_state(self, batch_size=1):
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def initial_blstats_state(self, batch_size=1):
        return torch.zeros(batch_size, self.bottomline_encoder.blstats_dim)

    @torch.no_grad()
    def update_running_moments(self, reward_batch):
        """Maintains a running mean of reward."""
        new_count = len(reward_batch)
        new_sum = torch.sum(reward_batch)
        new_mean = new_sum / new_count

        curr_mean = self.reward_sum / self.reward_count
        new_m2 = torch.sum((reward_batch - new_mean) ** 2) + (
            (self.reward_count * new_count)
            / (self.reward_count + new_count)
            * (new_mean - curr_mean) ** 2
        )

        self.reward_count += new_count
        self.reward_sum += new_sum
        self.reward_m2 += new_m2

    @torch.no_grad()
    def get_running_std(self):
        """Returns standard deviation of the running mean of the reward."""
        return torch.sqrt(self.reward_m2 / self.reward_count)

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def _embed_actions(self, lagged_actions, T, B):
        lags = lagged_actions.view(T * B, -1).long()
        action_emb = []

        for i in range(lags.shape[1]):
            try:
                action_emb.append(self._select(self.action_embed,
                                  lags[:, i]))
            except:
                logging.info(lags.shape)
                logging.info(lags[:, i].min())
                logging.info(lags[:, i].max())

        # -- [B x W x H x K]
        action_rep = torch.cat(action_emb, dim=-1)#self.extract_action_representation(action_emb)
        action_rep = action_rep.view(T * B, -1)
        return action_rep

    @torch.no_grad()
    def update_running_moments(self, reward_batch):
        """Maintains a running mean of reward."""
        new_count = len(reward_batch)
        new_sum = torch.sum(reward_batch)
        new_mean = new_sum / new_count

        curr_mean = self.reward_sum / self.reward_count
        new_m2 = torch.sum((reward_batch - new_mean) ** 2) + (
            (self.reward_count * new_count)
            / (self.reward_count + new_count)
            * (new_mean - curr_mean) ** 2
        )

        self.reward_count += new_count
        self.reward_sum += new_sum
        self.reward_m2 += new_m2

    @torch.no_grad()
    def get_running_std(self):
        """Returns standard deviation of the running mean of the reward."""
        return torch.sqrt(self.reward_m2 / self.reward_count)

    def forward(self, env_outputs, core_state):
        T, B, C, H, W = env_outputs["tty_chars"].shape

        # Take last channel for now
        topline = env_outputs["tty_chars"][..., -1, 0, :].contiguous()
        bottom_line = env_outputs["tty_chars"][..., -1, -2:, :].contiguous()

        # Blstats
        if self.use_parsed_blstats:
            blstats_rep = self.bottomline_encoder(
                env_outputs['blstats'].float(memory_format=torch.contiguous_format).view(T * B, -1),
                bottom_line.float(memory_format=torch.contiguous_format),
                env_outputs['roles'].float(memory_format=torch.contiguous_format).view(T * B) if self.use_bl_role else None
            )
        else:
            blstats_rep = self.bottomline_encoder(
                bottom_line.float(memory_format=torch.contiguous_format).view(T * B, -1)
            ) 

        st = [
            self.topline_encoder(
                topline.float(memory_format=torch.contiguous_format).view(T * B, -1)
            ),
            blstats_rep
        ]

        # Main obs encoding
        tty_chars = env_outputs["tty_chars"][..., 1:-2, :].contiguous().float(memory_format=torch.contiguous_format).view(T * B, C, H - 3, W)
        tty_colors = env_outputs["tty_colors"][..., 1:-2, :].contiguous().view(T * B, C, H - 3, W)
        tty_cursor = env_outputs["tty_cursor"].contiguous().view(T * B, -1)
        st.append(self.screen_encoder(tty_chars, tty_colors))

        # Previous action encoding
        if self.use_prev_action:
            st.append(
                torch.nn.functional.one_hot(
                    env_outputs["prev_action"], self.num_actions
                ).view(T * B, -1)
            )

        if self.use_cur_action:
            st.append(
                torch.nn.functional.one_hot(
                    env_outputs["cur_action"], self.num_actions
                ).view(T * B, -1)
            )

        if self.lagged_actions:
            lagged_actions = env_outputs['lagged_actions'].contiguous()
            action_rep = self._embed_actions(lagged_actions, T, B)
            assert action_rep.shape[0] == T * B
            st.append(action_rep)

        # Crop encoding
        if self.use_crop:
            tty_cursor = tty_cursor.clone() # very important! otherwise we'll mess with tty_cursor below
            tty_cursor[:, 0] -= 1 # adjust y position for cropping below
            tty_cursor = tty_cursor.flip(-1) # flip (y, x) to be (x, y)
            crop_tty_chars = self.crop(tty_chars[..., -1, :, :], tty_cursor)
            crop_tty_colors = self.crop(tty_colors[..., -1, :, :], tty_cursor)
            crop_chars = selectt(self.screen_encoder.char_embeddings, crop_tty_chars.long(), True)
            crop_colors = selectt(self.screen_encoder.color_embeddings, crop_tty_colors.long(), True)
            crop_obs = torch.cat([crop_chars, crop_colors], dim=-1)
            st.append(
                self.extract_crop_representation(crop_obs.permute(0, 3, 1, 2).contiguous()).view(T * B, -1)
            )

        st = torch.cat(st, dim=1)

        if self.use_policy_and_value_heads:
            st = self.fc(st)
            if self.use_lstm:
                core_input = st.view(T, B, -1)
                core_output_list = []
                notdone = (~env_outputs["done"]).float()

                for input, nd in zip(core_input.unbind(), notdone.unbind()):
                    # Reset core state to zero whenever an episode ended.
                    # Make `done` broadcastable with (num_layers, B, hidden_size)
                    nd = nd.view(1, -1, 1)
                    core_state = tuple(nd * t for t in core_state)
                    output, core_state = self.core(input.unsqueeze(0), core_state)
                    core_output_list.append(output)

                core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
            else:
                core_output = st

            # -- [B' x A]
            policy_logits = self.policy(core_output)

            # -- [B' x 1]
            if self.il_mode:
                baseline = torch.zeros(*policy_logits.shape[:1])
            else:
                baseline = self.baseline(core_output)

            if self.sampling_type == 'softmax':
                policy_logits *= self.temperature
                action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
            elif self.sampling_type == 'argmax':
                action = torch.argmax(policy_logits, dim=1).unsqueeze(1)
                policy_logits *= 1e9 # just something big enough here
            elif self.sampling_type == 'topp':
                policy_logits = self.top_p_logits_warper(None, policy_logits * self.temperature)
                action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
            elif self.sampling_type == 'topk':
                policy_logits = self.top_k_logits_warper(None, policy_logits * self.temperature)
                action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
            else:
                raise Exception('Unexpected sampling type!')

            policy_logits = policy_logits.view(T, B, -1)
            baseline = baseline.view(T, B)
            action = action.view(T, B)

            output = dict(
                policy_logits=policy_logits,
                baseline=baseline,
                action=action
            )

            return (output, core_state)
        else:
            return dict(
                state=st
            )

class CharColorEncoderResnet(nn.Module):
    """
    Inspired by network from IMPALA https://arxiv.org/pdf/1802.01561.pdf
    """
    def __init__(
        self, 
        screen_shape, 
        h_dim: int = 512, 
        scale_cnn_channels: int = 1, 
        num_fc_layers: int = 1, 
        char_edim: int = 16, 
        color_edim: int = 16,
        obs_frame_stack: int = 1,
        num_res_blocks: int = 2,
        num_res_layers: int = 2,
        kernel_size: int = 3,
        no_max_pool: bool = False,
        screen_conv_blocks: int = 3
    ):
        super(CharColorEncoderResnet, self).__init__()

        self.h, self.w = screen_shape
        self.h_dim = h_dim
        self.num_fc_layers = num_fc_layers
        self.char_edim = char_edim 
        self.color_edim = color_edim
        self.no_max_pool = no_max_pool
        self.screen_conv_blocks = screen_conv_blocks

        self.blocks = []

        self.conv_params = [
            [char_edim * obs_frame_stack + color_edim * obs_frame_stack, int(16 * scale_cnn_channels), kernel_size, num_res_blocks],
            [int(16 * scale_cnn_channels), int(32 * scale_cnn_channels), kernel_size, num_res_blocks],
            [int(32 * scale_cnn_channels), int(32 * scale_cnn_channels), kernel_size, num_res_blocks]
        ]

        self.conv_params = self.conv_params[:self.screen_conv_blocks]

        for (
            in_channels,
            out_channels,
            filter_size,
            num_res_blocks
        ) in self.conv_params:
            block = []
            # Downsample
            block.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    filter_size,
                    stride=1,
                    padding=(filter_size // 2)
                )
            )
            if not self.no_max_pool:
                block.append(
                    nn.MaxPool2d(
                        kernel_size=3,
                        stride=2
                    )
                )
                self.h = math.floor((self.h - 1 * (3 - 1) - 1)/2 + 1) # from PyTorch Docs
                self.w = math.floor((self.w - 1 * (3 - 1) - 1)/2 + 1) # from PyTorch Docs

            # Residual block(s)
            for _ in range(num_res_blocks):
                block.append(ResBlock(out_channels, out_channels, filter_size, num_res_layers))
            self.blocks.append(nn.Sequential(*block))

        self.conv_net = nn.Sequential(*self.blocks)
        self.out_size = self.h * self.w * out_channels

        fc_layers = [nn.Linear(self.out_size, self.h_dim), nn.ELU(inplace=True)]
        for _ in range(self.num_fc_layers - 1):
            fc_layers.append(nn.Linear(self.h_dim, self.h_dim))
            fc_layers.append(nn.ELU(inplace=True))
        self.fc_head = nn.Sequential(*fc_layers)

        self.char_embeddings = nn.Embedding(256, self.char_edim)
        self.color_embeddings = nn.Embedding(128, self.color_edim)

    def forward(self, chars, colors):
        chars, colors = self._embed(chars, colors) # 21 x 80
        x = self._stack(chars, colors)
        x = self.conv_net(x)
        x = x.view(-1, self.out_size)
        x = self.fc_head(x)
        return x

    def _embed(self, chars, colors):
        chars = selectt(self.char_embeddings, chars.long(), True)
        colors = selectt(self.color_embeddings, colors.long(), True)
        return chars, colors

    def _stack(self, chars, colors):
        obs = torch.cat([chars, colors], dim=-1)
        return obs.permute(0, 1, 4, 2, 3).flatten(1, 2).contiguous()

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, filter_size: int, num_layers: int):
        super(ResBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    filter_size,
                    stride=1,
                    padding=(filter_size // 2)
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ELU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) + x

class BottomLinesEncoder(nn.Module):
    """
    Adapted from https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022/blob/67139262966aa11555cf7aca15723375b36fbe42/experiment_code/hackrl/models/offline_chaotic_dwarf.py
    """
    def __init__(self, h_dim: int = 128, scale_cnn_channels: int = 1):
        super(BottomLinesEncoder, self).__init__()
        self.conv_layers = []
        w = nethack.NLE_TERM_CO * 2
        for in_ch, out_ch, filter, stride in [[2, int(32 * scale_cnn_channels), 8, 4], [int(32 * scale_cnn_channels), int(64 * scale_cnn_channels), 4, 1]]:
            self.conv_layers.append(nn.Conv1d(in_ch, out_ch, filter, stride=stride))
            self.conv_layers.append(nn.ELU(inplace=True))
            w = conv_outdim(w, filter, padding=0, stride=stride)

        self.h_dim = h_dim

        self.out_dim = w * out_ch
        self.conv_net = nn.Sequential(*self.conv_layers)
        self.fwd_net = nn.Sequential(
            nn.Linear(self.out_dim, self.h_dim),
            nn.ELU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ELU(),
        )

    def forward(self, bottom_lines):
        B, D = bottom_lines.shape
        # ASCII 32: ' ', ASCII [33-128]: visible characters
        chars_normalised = (bottom_lines - 32) / 96

        # ASCII [45-57]: -./01234556789
        numbers_mask = (bottom_lines > 44) * (bottom_lines < 58)
        digits_normalised = numbers_mask * (bottom_lines - 47) / 10 # why subtract 47 here and not 48?

        # Put in different channels & conv (B, 2, D)
        x = torch.stack([chars_normalised, digits_normalised], dim=1)
        return self.fwd_net(self.conv_net(x).view(B, -1))

class TopLineEncoder(nn.Module):
    """
    This class uses a one-hot encoding of the ASCII characters
    as features that get fed into an MLP. 
    Adapted from https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022/blob/67139262966aa11555cf7aca15723375b36fbe42/experiment_code/hackrl/models/offline_chaotic_dwarf.py
    """
    def __init__(self, msg_hdim: int):
        super(TopLineEncoder, self).__init__()
        self.msg_hdim = msg_hdim
        self.i_dim = nethack.NLE_TERM_CO * 256

        self.msg_fwd = nn.Sequential(
            nn.Linear(self.i_dim, self.msg_hdim),
            nn.ELU(inplace=True),
            nn.Linear(self.msg_hdim, self.msg_hdim),
            nn.ELU(inplace=True),
        )


    def forward(self, message):
        # Characters start at 33 in ASCII and go to 128. 96 = 128 - 32
        message_normed = (
            F.one_hot((message).long(), 256).reshape(-1, self.i_dim).float()
        )
        return self.msg_fwd(message_normed)

def conv_outdim(i_dim, k, padding=0, stride=1, dilation=1):
    """Return the dimension after applying a convolution along one axis"""
    return int(1 + (i_dim + 2 * padding - dilation * (k - 1) - 1) / stride)

def selectt(embedding_layer, x, use_index_select):
    """Use index select instead of default forward to possible speed up embedding."""
    if use_index_select:
        out = embedding_layer.weight.index_select(0, x.view(-1))
        # handle reshaping x to 1-d and output back to N-d
        return out.view(x.shape + (-1,))
    else:
        return embedding_layer(x)