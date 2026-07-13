# origin: https://github.com/NiftyliuS/goblin-model
from transformers import PreTrainedConfig


class GoblinConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GoblinModel`]. It is used to instantiate a Goblin
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Goblin model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GoblinModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 16384):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Goblin Block.

        attn_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Goblin Block.
        attn_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attn_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.

        conv_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the 1D causal convolution (`nn.Conv1d`) layer that blends adjacent tokens.
        conv_k (`int`, *optional*, defaults to 4):
            The kernel size for the 1D causal convolution (`nn.Conv1d`) layer. Acts as a dynamic n-gram
            extractor, permanently fusing strict local concepts (like operators and operands) into a
            translation-invariant representation before the global attention routing evaluates them.

        mlp_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the Goblin Block.
        mlp_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.

        out_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the logits generation layer.

        out_head_dim (`int`, *optional*, defaults to 3):
            The number of independent spatial dimensions (coordinates) a single categorical target token is mathematically
            decomposed into. Instead of predicting a flat 1D vector across the entire vocabulary, the model predicts discrete
            coordinates on a multi-dimensional grid. Summing the categorical cross-entropies of these dimensions provides dense,
            stable orthogonal gradient signals while entirely bypassing the catastrophic gradient collapse associated with
            continuous MSE regression.

        out_heads (`int`, *optional*, defaults to 2):
            The number of parallel, independent routing ensembles (heads) used for token prediction. Each head maps the target
            vocabulary to a completely distinct, randomly permuted coordinate grid (`torch.randperm`). This forces the network
            to learn generalized operational logic rather than memorizing numerical index proximity. During loss calculation,
            the sum is divided by this value to compute the average ensemble loss, perfectly normalizing the final gradient
            magnitude to match standard single-head architectures.

        g_size (`int`, *optional*, defaults to None):
            The length of a single coordinate axis (grid resolution) for the multi-dimensional token routing. If left as `None`,
            it is dynamically calculated as `ceil(vocab_size ** (1 / out_head_dim))` to strictly envelop the mathematical volume
            of the vocabulary.

        rms_norm_eps (`float`, *optional*, defaults to None):
            The epsilon used by the rms normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to
            understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
            results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).

        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pad_token_id (`int`, *optional*, defaults to 2):
            Padding token id.
    """

    model_type = "goblin"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `GoblinModel`
    base_model_tp_plan = {
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "causal_conv": (["hidden_states"], ["hidden_states"]),
        "layers": (["x", "attn_mask"], ["x"]),
        "out_norm": (["input"], ["output"]),
    }

    def __init__(
            self,
            vocab_size: int | None = 32000,
            vocab_blocks: int | None = None,
            hidden_size: int | None = 4096,
            intermediate_size: int | None = 16384,
            num_hidden_layers: int | None = 32,

            attn_heads: int | None = 32,
            attn_dropout: float | None = 0.0,
            attn_bias: bool | None = True,
            conv_bias: bool | None = False,
            conv_k: int | None = 4,

            mlp_act: str | None = "silu",
            mlp_bias: bool | None = True,

            out_bias: bool | None = False,
            out_head_dim: int | None = 2,
            out_heads: int | None = 2,
            g_size: int | None = None,

            rms_norm_eps: float | None = None,
            initializer_range: float | None = 0.02,
            pretraining_tp: int | None = 1,

            bos_token_id: int | None = 1,
            eos_token_id: int | None = 2,
            pad_token_id: int | None = 2,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.vocab_blocks = vocab_blocks

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.mlp_act = mlp_act
        self.mlp_bias = mlp_bias

        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.attn_bias = attn_bias
        self.conv_bias = conv_bias
        self.conv_k = conv_k

        self.out_bias = out_bias
        self.out_head_dim = out_head_dim
        self.out_heads = out_heads
        self.g_size = g_size

        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.pretraining_tp = pretraining_tp

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # unsupported ( TBD )
        self.use_cache = False
        super().__init__(use_cache=False, **kwargs)


__all__ = ["GoblinConfig"]
