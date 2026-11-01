"""Schema of the published predictor's config.json.

The file describes a bank of per-layer heads that predict, at the end of token t,
which routed experts each target layer will use for token t+1. A runtime must refuse
a file whose numbers do not match the base model it serves: the heads are trained on
one model's hidden states and routing, and silent reuse on another model gives
meaningless probabilities, not an error.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FORMAT_VERSION = 1
# Format version 1 fixes the order of the input blocks. The heads were trained on this
# order, so a config with other names or another order is a different format and fails.
V1_BLOCKS = ('h', 'win', 'prior', 'pos', 'rl', 'prevwin', 'hd', 'sticky')


class Frozen(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class BaseModelSpec(Frozen):
    repo_id: str
    revision: str = Field(pattern=r'^[0-9a-f]{40}$')
    num_layers: int = Field(gt=0)
    num_experts: int = Field(gt=0)
    top_k: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    vocab_size: int = Field(gt=0)

    @model_validator(mode='after')
    def _top_k_fits(self):
        if self.top_k > self.num_experts:
            raise ValueError(f'top_k {self.top_k} > num_experts {self.num_experts}')
        return self


class FeatureBlock(Frozen):
    """One block of the per-head input vector, in the order the heads concatenate them."""
    name: str
    width: int = Field(gt=0)
    meaning: str


class FeatureSpec(Frozen):
    clock: Literal['end_of_token_t']
    forecast: Literal['token_t_plus_1']
    # Layer offsets are relative to the head's target layer L.
    hidden_source_offset: int = Field(le=0)       # h = MoE input hidden state of layer L + offset
    route_window: int = Field(gt=0)               # token t's routed experts at layers L-1 .. L-route_window
    previous_token_window: int = Field(gt=0)      # token t-1's routed experts at layers L-1 .. L-window
    router_logit_window: int = Field(gt=0)        # token t's router logits at layers L-1 .. L-window
    router_logit_layers_zeroed: tuple[int, ...]   # layers whose logits the deployed tracker never stored
    ema_beta: float = Field(gt=0, lt=1)           # prior <- beta * prior, then + (1 - beta) at token t's experts
    position_scale: int = Field(gt=0)             # position feature = (decode step + 1) / position_scale
    token_embedding_dim: int = Field(gt=0)
    # Rows of the frozen token table (a seeded random projection, not model weights). The
    # table covers the text vocabulary, ids 0 .. rows-1. The tokenizer also defines audio and
    # TTS tokens above it (248070-248076 for the published base model), and the model's
    # vocab_size includes padding. A token id at or above this value has no row: a runtime
    # must reject it, never invent a row for it.
    token_table_rows: int = Field(gt=0)
    blocks: tuple[FeatureBlock, ...]


class HeadArchitecture(Frozen):
    """Shapes of one head; every head in the bank has the same shapes."""
    feature_dim: int = Field(gt=0)      # sum of FeatureSpec.blocks widths
    forecast_hidden: int = Field(gt=0)  # fh_down output: predicts the change of the hidden state
    proj_dim: int = Field(gt=0)
    up_dim: int = Field(gt=0)
    activation: Literal['gelu_erf']


class ServingReference(Frozen):
    """The configuration the published numbers were measured with. Informational only."""
    engine: str
    engine_commit: str
    precision: Literal['int8_group128', 'bf16']
    int8_group: int = Field(gt=0)
    cache_slots: int = Field(gt=0)
    kv_pages: int = Field(gt=0)
    section_caps: tuple[int, ...]
    demand_copy_stripes: int = Field(gt=0)
    first_prefetch_step: int = Field(ge=0)


class PredictorConfig(Frozen):
    format_version: Literal[1]
    base_model: BaseModelSpec
    target_layers: tuple[int, ...]  # head i scores target_layers[i]
    features: FeatureSpec
    architecture: HeadArchitecture
    calibration: Literal['sigmoid(a * score + b), per head']
    serving_reference: ServingReference
    checkpoint_sha256: dict[str, str]  # per target layer: the training checkpoint the head came from
    source_commit: str

    @model_validator(mode='after')
    def _consistent(self):
        layers, base, feats = self.target_layers, self.base_model, self.features
        if list(layers) != sorted(set(layers)):
            raise ValueError('target_layers must be strictly increasing')
        # Every window must reach only layers that exist; layer L-w < 0 has no state.
        reach = max(feats.route_window, feats.previous_token_window, feats.router_logit_window,
                    -feats.hidden_source_offset + 1)
        if not layers or layers[0] - reach < 0 or layers[-1] >= base.num_layers:
            raise ValueError(f'target layers {layers[0]}..{layers[-1]} need layers down to '
                             f'{layers[0] - reach}, outside 0..{base.num_layers - 1}')
        names = tuple(b.name for b in feats.blocks)
        if names != V1_BLOCKS:
            raise ValueError(f'feature blocks {names}, format version 1 requires {V1_BLOCKS}')
        e, d = base.num_experts, base.hidden_size
        implied = dict(h=d, win=feats.route_window * e, prior=e, pos=1, rl=feats.router_logit_window * e,
                       prevwin=feats.previous_token_window * e, hd=d, sticky=e)
        for b in feats.blocks:
            if b.width != implied[b.name]:
                raise ValueError(f'feature block {b.name}: width {b.width}, the windows and base model imply '
                                 f'{implied[b.name]}')
        width = sum(b.width for b in feats.blocks)
        if width != self.architecture.feature_dim:
            raise ValueError(f'feature blocks sum to {width}, architecture says {self.architecture.feature_dim}')
        if feats.token_table_rows > base.vocab_size:
            raise ValueError(f'token table has {feats.token_table_rows} rows, model vocab is {base.vocab_size}')
        if set(self.checkpoint_sha256) != {str(layer) for layer in layers}:
            raise ValueError('checkpoint_sha256 must name exactly the target layers')
        if len(self.serving_reference.section_caps) == 0:
            raise ValueError('section_caps is empty')
        return self

    def check_base(self, hf_config: dict) -> None:
        """Raise unless a Hugging Face config.json (or its text_config) has the base model's shapes.

        Shapes only: a fine-tune of the same architecture passes this check and still gives
        meaningless probabilities. The caller must also pin base_model.repo_id and revision.
        """
        cfg = hf_config.get('text_config', hf_config)
        want = dict(num_layers=('num_hidden_layers',), num_experts=('num_experts',),
                    top_k=('num_experts_per_tok',), hidden_size=('hidden_size',),
                    vocab_size=('vocab_size',))
        for field, keys in want.items():
            got = next((cfg[k] for k in keys if k in cfg), None)
            if got != getattr(self.base_model, field):
                raise ValueError(f'base model {field}: predictor expects '
                                 f'{getattr(self.base_model, field)}, model config has {got}')
