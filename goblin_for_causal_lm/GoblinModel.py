# origin: https://github.com/NiftyliuS/goblin-model
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.activations import ACT2FN

from .GoblinConfig import GoblinConfig

logger = logging.get_logger(__name__)


class GoblinMLP(nn.Module):
    def __init__(self, config: GoblinConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GoblinBlock(nn.Module):
    def __init__(self, config: GoblinConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.attn_heads,
            dropout=config.attn_dropout,
            bias=config.attn_bias,
            batch_first=True
        )
        self.mlp = GoblinMLP(config)

    def forward(self, x, attn_mask):
        x_i = self.attn_norm(x)
        a, _ = self.attn(x_i, x_i, x_i, attn_mask=attn_mask, need_weights=False)
        # non standard approach prevents attention bleed into the residual stream forcing the block to be contained
        x = x + self.mlp(self.mlp_norm(a))
        return x


class GoblinModel(PreTrainedModel):
    config_class = GoblinConfig
    _no_split_modules = ["GoblinBlock"]

    def __init__(self, config: GoblinConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.num_heads = config.attn_heads
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # We replace the heavy attention with a lightweight Depthwise Causal Conv.
        # This blends token 't' with 't-1' (e.g., mixes the variable with the preceding operator)
        # Highly parameter-efficient (only `hidden_size` parameters), perfect for grokking.
        self.causal_conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_k,
            padding=config.conv_k - 1,  # Causal padding prevents looking into the future
            groups=config.hidden_size,  # Depthwise ensures channels don't cross-contaminate
            bias=config.conv_bias
        )

        self.layers = nn.ModuleList([GoblinBlock(config) for _ in range(config.num_hidden_layers)])
        self.out_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            output_hidden_states: bool | None = None,
            **kwargs,
    ) -> BaseModelOutputWithPast:

        # Resolve dynamically: use passed value, otherwise fallback to config
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        seq_len = h.shape[1]
        batch_size = h.shape[0]

        is_pad = None
        if attention_mask is not None:
            is_pad = (attention_mask == 0)
            h = h.masked_fill(is_pad.unsqueeze(-1), 0.0)

        # ---------------------------------------------------------
        # ALiBi / Multi-Head Decay Mask Generation
        # ---------------------------------------------------------
        seq_range = torch.arange(seq_len, device=h.device)
        distance_matrix = seq_range.unsqueeze(1) - seq_range.unsqueeze(0)
        causal_bool = distance_matrix < 0

        # Paper-Accurate ALiBi Slopes - generates a geometric sequence spanning from 2^(-8/num_heads) down to 2^-8
        slopes = torch.pow(
            2.0, -torch.arange(1, self.num_heads + 1, dtype=torch.float32, device=h.device) * (8.0 / self.num_heads)
        ).to(h.dtype)

        attn_mask_float = -slopes.view(self.num_heads, 1, 1) * distance_matrix.to(h.dtype)
        attn_mask_float.masked_fill_(causal_bool, float('-inf'))
        attn_mask_float = attn_mask_float.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()

        if is_pad is not None:
            attn_mask_float.masked_fill_(is_pad.unsqueeze(1).unsqueeze(2), float('-inf'))
            diag = torch.eye(seq_len, dtype=torch.bool, device=h.device).unsqueeze(0).unsqueeze(0)
            pad_query_rows = is_pad.unsqueeze(1).unsqueeze(3)
            attn_mask_float.masked_fill_(pad_query_rows & diag, 0.0)

        attn_mask_float = attn_mask_float.view(batch_size * self.num_heads, seq_len, seq_len)
        # ---------------------------------------------------------

        h_conv = h.transpose(1, 2)
        h_conv = self.causal_conv(h_conv)
        h_conv = h_conv[..., :seq_len]
        h = h_conv.transpose(1, 2)

        all_hidden_states = () if output_hidden_states else None

        if output_hidden_states:
            all_hidden_states += (h,)

        for layer in self.layers:
            h = layer(h, attn_mask_float)
            if output_hidden_states:
                all_hidden_states += (h,)

        h = self.out_norm(h)
        return BaseModelOutputWithPast(
            last_hidden_state=h,
            hidden_states=all_hidden_states
        )
