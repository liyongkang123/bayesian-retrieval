from dataclasses import dataclass
from typing import Literal, Optional

from transformers import TrainingArguments as TR_ARGS


@dataclass
class ModelArguments:
    model_name_or_path: str
    model_type: Literal["vanilla", "vi", "mc_dropout"] = "vanilla"
    weight_sharing: Optional[bool] = False
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None


@dataclass
class TrainingArguments(TR_ARGS):
    prior_sigma: float = 1.0
    kld: float = 1e-6
    bias: Literal["informative", "uninformative"] = "informative"
    nsamples: int = 100
