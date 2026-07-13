# origin: https://github.com/NiftyliuS/goblin-model
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple, logging

from .GoblinConfig import GoblinConfig
from .GoblinModel import GoblinModel

logger = logging.get_logger(__name__)


class GoblinForCausalLM(PreTrainedModel):
    config_class = GoblinConfig
    _no_split_modules = ["GoblinBlock"]

    def __init__(self, config: GoblinConfig):
        super().__init__(config)
        self.model = GoblinModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.out_bias)

        self.post_init()

    @can_return_tuple
    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            output_hidden_states: bool | None = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
