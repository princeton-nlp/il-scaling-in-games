# Standard library imports
import logging
import re
import math
from typing import Optional, List

# Third party imports
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Model, GPT2Config, LlamaConfig, LlamaModel
from nle import nethack  # noqa: E402
from nle.nethack.nethack import TERMINAL_SHAPE
from transformers import TopPLogitsWarper, TopKLogitsWarper
from mamba_ssm.models.config_mamba import MambaConfig
from omegaconf import DictConfig

# Local application imports
from il_scale.nethack.v2.networks.bottomline_net import BottomLineEncoder
from il_scale.nethack.v2.networks.topline_net import TopLineEncoder
from il_scale.nethack.v2.networks.policy_head import PolicyHead
from il_scale.nethack.v2.networks.modality_mixer import ModalityMixer
from il_scale.nethack.v2.networks.resnet import CharColorEncoderResnet
from il_scale.nethack.v2.networks.inv_net import InventoryNet
from il_scale.nethack.v2.networks.crop_net import Crop
from il_scale.nethack.v2.networks.mamba_net import MambaCore
from il_scale.nethack.v2.utils.model import selectt, interleave

# A logger for this file
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=logging.INFO,
)

class PolicyNet(nn.Module):
    """
    Main NetHack net.
    """
    def __init__(self, cfg: DictConfig):
        super(PolicyNet, self).__init__()

        self.cfg = cfg
        self.sampling_type = self.cfg.rollout.sampling_type
        self.temperature = self.cfg.rollout.temperature
        self.obs_shape = TERMINAL_SHAPE

        # Register buffers
        self.register_buffer("reward_sum", torch.zeros(()))
        self.register_buffer("reward_m2", torch.zeros(()))
        self.register_buffer("reward_count", torch.zeros(()).fill_(1e-8))

        # Sampling utilities
        self.top_p_logits_warper = TopPLogitsWarper(self.cfg.rollout.top_p)
        self.top_k_logits_warper = TopKLogitsWarper(self.cfg.rollout.top_k)
    
        # Register cropping nets
        # NOTE: -3 because we cut the topline and bottom two lines
        if not self.cfg.network.include_top_and_bottom:
            self.crop = Crop(self.obs_shape[0] - 3, self.obs_shape[1], self.cfg.network.crop_dim, self.cfg.network.crop_dim)
        else:
            self.crop = Crop(self.obs_shape[0], self.obs_shape[1], self.cfg.network.crop_dim, self.cfg.network.crop_dim)
        crop_in_channels = ([self.cfg.network.char_edim + self.cfg.network.color_edim] if not self.cfg.network.add_char_color else [self.cfg.network.char_edim]) + [self.cfg.network.crop_inter_filters] * (self.cfg.network.crop_num_layers - 1)
        crop_out_channels = [self.cfg.network.crop_inter_filters] * (self.cfg.network.crop_num_layers - 1) + [self.cfg.network.crop_out_filters]
        conv_extract_crop = []
        norm_extract_crop = []
        for i in range(self.cfg.network.crop_num_layers):
            conv_extract_crop.append(nn.Conv2d(
                in_channels=crop_in_channels[i],
                out_channels=crop_out_channels[i],
                kernel_size=(self.cfg.network.crop_kernel_size, self.cfg.network.crop_kernel_size),
                stride=self.cfg.network.crop_stride,
                padding=self.cfg.network.crop_padding,
            ))
            norm_extract_crop.append(nn.BatchNorm2d(crop_out_channels[i]))
        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, norm_extract_crop, [nn.ELU()] * len(conv_extract_crop))
        )
        self.crop_out_dim = self.cfg.network.crop_dim ** 2 * self.cfg.network.crop_out_filters

        # Top and bottomline encoders
        self.topline_encoder = TopLineEncoder(cfg, msg_hdim=self.cfg.network.msg_hdim)
        self.bottomline_encoder = BottomLineEncoder(cfg, hdim=self.cfg.network.blstats_hdim//4)

        # Inventory encoders
        if self.cfg.network.use_inventory:
            self.inventory_encoder = InventoryNet(inv_edim=self.cfg.network.inv_edim, inv_hdim=self.cfg.network.inv_hdim)

        # Register main observation encoder
        obs_shape = (self.obs_shape[0] - 3, self.obs_shape[1]) if not self.cfg.network.include_top_and_bottom else self.obs_shape
        self.obs_encoder = CharColorEncoderResnet(obs_shape, self.cfg)

        self.num_actions = len(nethack.ACTIONS)
        self.prev_actions_dim = self.num_actions

        self.out_dim = sum(
            [
                self.topline_encoder.msg_hdim,
                self.bottomline_encoder.hdim,
                self.obs_encoder.hdim,
                self.prev_actions_dim,
                self.crop_out_dim,
                self.inventory_encoder.inv_hdim if self.cfg.network.use_inventory else 0
            ]
        )

        # Register sequence model on top
        if self.cfg.network.core_mode == 'mamba':
            mamba_config = MambaConfig(
                d_model=self.cfg.network.hdim,
                n_layer=self.cfg.network.mamba_num_layers
            )
            print('num mamba layers:', mamba_config.n_layer)
            print('mamba hidden size:', mamba_config.d_model)
            self.core = MambaCore(mamba_config)

        elif self.cfg.network.core_mode == 'gpt2':
            gpt2config = GPT2Config(
                n_embd=self.cfg.network.hdim, 
                n_layer=self.cfg.network.tf_num_layers, 
                n_head=self.cfg.network.tf_num_heads,
                vocab_size=1, # we feed in our own embeddings, 0 gives DDP error
                n_positions=self.cfg.data.unroll_length + 1
            )
            self.core = GPT2Model(gpt2config)
            print('gpt2 hidden size:', gpt2config.n_embd)
            print('gpt2 num layers:', gpt2config.n_layer)

        elif self.cfg.network.core_mode == 'llama':
            llama_config = LlamaConfig(
                vocab_size=1,
                hidden_size=self.cfg.network.hdim,
                num_hidden_layers=self.cfg.network.tf_num_layers,
                num_attention_heads=self.cfg.network.tf_num_heads,
                max_position_embeddings=self.cfg.data.unroll_length +1,
                intermediate_size=4 * self.cfg.network.hdim
            )

            self.core = LlamaModel(llama_config)
            print('llama hidden size:', llama_config.hidden_size)
            print('llama num layers:', llama_config.num_hidden_layers)

        self.modality_mixer = ModalityMixer(cfg, self.out_dim)
        self.policy_head = PolicyHead(cfg, self.num_actions)

    def forward(self, env_outputs, inference_params = None):
        T, B, C, H, W = env_outputs["tty_chars"].shape

        # Encode blstats 
        bottom_line = env_outputs["tty_chars"][..., -1, -2:, :].contiguous()
        blstats_rep = self.bottomline_encoder(
            bottom_line.float(memory_format=torch.contiguous_format).view(T * B, -1)
        ) 

        # Encode topline
        topline = env_outputs["tty_chars"][..., -1, 0, :].contiguous()
        topline_rep = self.topline_encoder(
            topline.float(memory_format=torch.contiguous_format).view(T * B, -1)
        )

        st = [topline_rep, blstats_rep]

        # Encode inventory if using
        if self.cfg.network.use_inventory:
            inv_rep = self.inventory_encoder(
                env_outputs["inv_glyphs"].contiguous().int().view(T * B, -1), 
            )

            st.append(inv_rep)

        # Encode main observation
        if not self.cfg.network.include_top_and_bottom:
            tty_chars = env_outputs["tty_chars"][..., 1:-2, :].contiguous().float(memory_format=torch.contiguous_format).view(T * B, C, H - 3, W)
            tty_colors = env_outputs["tty_colors"][..., 1:-2, :].contiguous().view(T * B, C, H - 3, W)
        else:
            tty_chars = env_outputs["tty_chars"].contiguous().float(memory_format=torch.contiguous_format).view(T * B, C, H, W)
            tty_colors = env_outputs["tty_colors"].contiguous().view(T * B, C, H, W)

        tty_cursor = env_outputs["tty_cursor"].contiguous().view(T * B, -1)
        st.append(self.obs_encoder(tty_chars, tty_colors))

        # Encode previous action
        st.append(
            torch.nn.functional.one_hot(
                env_outputs["prev_action"], self.num_actions
            ).view(T * B, -1)
        )

        # Encode crop 
        tty_cursor = tty_cursor.clone() # very important! otherwise we'll mess with tty_cursor below

        if not self.cfg.network.include_top_and_bottom:
            tty_cursor[:, 0] -= 1 # adjust y position for cropping below

        tty_cursor = tty_cursor.flip(-1) # flip (y, x) to be (x, y)
        crop_tty_chars = self.crop(tty_chars[..., -1, :, :], tty_cursor)
        crop_tty_colors = self.crop(tty_colors[..., -1, :, :], tty_cursor)
        crop_chars = selectt(self.obs_encoder.char_embeddings, crop_tty_chars.long(), True)
        crop_colors = selectt(self.obs_encoder.color_embeddings, crop_tty_colors.long(), True)

        if self.cfg.network.add_char_color:
            crop_obs = crop_chars + crop_colors
        else:
            crop_obs = torch.cat([crop_chars, crop_colors], dim=-1)
        st.append(
            self.extract_crop_representation(crop_obs.permute(0, 3, 1, 2).contiguous()).view(T * B, -1)
        )

        st = torch.cat(st, dim=1)
        st = self.modality_mixer(st)

        # Send through sequence model
        core_input = st.view(T, B, -1).transpose(0, 1).contiguous()
        if self.cfg.network.core_mode == 'mamba':
            core_output = self.core(core_input, inference_params=inference_params)

        elif self.cfg.network.core_mode == 'gpt2' or self.cfg.network.core_mode == 'llama':
            core_output = self.core(inputs_embeds=core_input)

        core_output = core_output.last_hidden_state        
        core_output = torch.flatten(core_output.transpose(0, 1).contiguous(), 0, 1)

        # Get policy logits
        policy_logits = self.policy_head(core_output)

        # Sampling
        if inference_params is not None:
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

            action = action.view(T, B)
        else:
            action = None

        policy_logits = policy_logits.view(T, B, -1)

        output = dict(
            policy_logits=policy_logits,
            action=action
        )

        return output

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