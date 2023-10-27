import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tevatron.modeling.encoder import EncoderOutput
from transformers import AutoModel

from bnir.bayesian_models import VariationalInferenceModel


class DenseEncoder(nn.Module):
    def __init__(self, qry_model, psg_model, weight_sharing):
        super().__init__()
        self.qry_model = qry_model
        self.psg_model = psg_model
        self.weight_sharing = weight_sharing

    def forward(self, query, passage):
        q_reps = self.qry_model(**query, return_dict=True).last_hidden_state[:, 0]
        p_reps = self.psg_model(**passage, return_dict=True).last_hidden_state[:, 0]

        scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        scores = scores.view(q_reps.size(0), -1)
        if self.training:
            target = torch.arange(
                scores.size(0), device=scores.device, dtype=torch.long
            )
            target = target * (p_reps.size(0) // q_reps.size(0))
            loss = F.cross_entropy(scores, target)
        else:
            loss = None

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @classmethod
    def build(cls, model_args, **hf_kwargs):
        qry_model = AutoModel.from_pretrained(
            model_args.model_name_or_path, **hf_kwargs
        )
        if model_args.weight_sharing:
            psg_model = qry_model
        else:
            psg_model = copy.deepcopy(qry_model)
        return cls(qry_model, psg_model, model_args.weight_sharing)

    @classmethod
    def load(cls, model_args, **hf_kwargs):
        if model_args.weight_sharing:
            qry_model = AutoModel.from_pretrained(
                model_args.model_name_or_path, **hf_kwargs
            )
            psg_model = qry_model
        else:
            qry_model = AutoModel.from_pretrained(
                os.path.join(model_args.model_name_or_path, "qry"), **hf_kwargs
            )
            psg_model = AutoModel.from_pretrained(
                os.path.join(model_args.model_name_or_path, "psg"), **hf_kwargs
            )
        return cls(qry_model, psg_model, model_args.weight_sharing)

    def save(self, output_dir):
        if self.weight_sharing:
            self.qry_model.save_pretrained(output_dir)
        else:
            self.qry_model.save_pretrained(os.path.join(output_dir, "qry"))
            self.psg_model.save_pretrained(os.path.join(output_dir, "psg"))


class BayesianDenseEncoder(nn.Module):
    def __init__(self, qry_model, psg_model, weight_sharing, prior_sigma, kld, ntrain):
        super().__init__()
        self.qry_model = qry_model
        self.psg_model = psg_model
        self.weight_sharing = weight_sharing
        self.prior_sigma = prior_sigma
        self.kld = kld
        self.ntrain = ntrain

    def _kl_div(self, base_model, prior_model, trained_model):
        kl_div = 0.0
        for p, p0, p_mu, p_sigma in zip(
            base_model.parameters(),
            prior_model.parameters(),
            trained_model.mu.parameters(),
            trained_model.sigma.parameters(),
        ):
            d = p.numel()
            sig2 = self.prior_sigma**2
            s = p_sigma.clamp(min=1e-8)
            v = s**2
            kl_div += 0.5 * (
                (((p_mu - p0) ** 2 + v) / sig2).sum() - (v / sig2).log().sum() - d
            )
        return kl_div

    def forward(
        self,
        query,
        passage,
        base_qry_model,
        base_psg_model,
        prior_model,
        nsamples=100,
    ):  # TODO: Use nsamples
        q_reps = self.qry_model(query, base_qry_model).last_hidden_state[:, 0]
        p_reps = self.psg_model(passage, base_psg_model).last_hidden_state[:, 0]

        scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        scores = scores.view(q_reps.size(0), -1)
        if self.training:
            target = torch.arange(
                scores.size(0), device=scores.device, dtype=torch.long
            )
            target = target * (p_reps.size(0) // q_reps.size(0))
            kl_div = self._kl_div(
                base_qry_model, prior_model, self.qry_model
            ) + self._kl_div(base_psg_model, prior_model, self.psg_model)
            loss = F.cross_entropy(scores, target) + self.kld * kl_div / self.ntrain
        else:
            loss = None

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @classmethod
    def build(cls, model_args, train_args, ntrain, **hf_kwargs):
        qry_model = VariationalInferenceModel(
            AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        )
        if model_args.weight_sharing:
            psg_model = qry_model
        else:
            psg_model = copy.deepcopy(qry_model)
        return cls(
            qry_model,
            psg_model,
            model_args.weight_sharing,
            train_args.prior_sigma,
            train_args.kld,
            ntrain,
        )

    @classmethod
    def load(cls, model_args, train_args, ntrain, **hf_kwargs):
        if model_args.weight_sharing:
            qry_model = AutoModel.from_pretrained(
                model_args.model_name_or_path, **hf_kwargs
            )
            psg_model = qry_model
        else:
            qry_model = AutoModel.from_pretrained(
                os.path.join(model_args.model_name_or_path, "qry"), **hf_kwargs
            )
            psg_model = AutoModel.from_pretrained(
                os.path.join(model_args.model_name_or_path, "psg"), **hf_kwargs
            )
        return cls(
            qry_model,
            psg_model,
            model_args.weight_sharing,
            train_args.prior_sigma,
            train_args.kld,
            ntrain,
        )

    def save(self, output_dir):
        if self.weight_sharing:
            self.qry_model.mu.save_pretrained(os.path.join(output_dir, "mu"))
            self.qry_model.sigma.save_pretrained(os.path.join(output_dir, "sigma"))
        else:
            self.qry_model.mu.save_pretrained(os.path.join(output_dir, "qry", "mu"))
            self.qry_model.sigma.save_pretrained(
                os.path.join(output_dir, "qry", "sigma")
            )
            self.psg_model.mu.save_pretrained(os.path.join(output_dir, "psg", "mu"))
            self.psg_model.sigma.save_pretrained(
                os.path.join(output_dir, "psg", "sigma")
            )
