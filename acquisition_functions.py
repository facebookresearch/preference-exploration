from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Optional

import torch
from botorch.acquisition import AcquisitionFunction, MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.gp_sampling import get_gp_samples
from botorch.utils.transforms import (concatenate_pending_points,
                                      match_batch_shape,
                                      t_batch_mode_transform)
from torch import Tensor
from torch.distributions import Bernoulli, Normal

from constants import *  # noqa: F403, F401
from helper_classes import LearnedPrefereceObjective, PosteriorMeanDummySampler


def get_rff_sample(outcome_model):
    om_without_transforms = deepcopy(outcome_model)
    om_without_transforms.input_transform = None
    om_without_transforms.outcome_transform = None

    gp_samples = get_gp_samples(
        model=om_without_transforms,
        num_outputs=om_without_transforms.num_outputs,
        n_samples=1,
        num_rff_features=500,
    )

    gp_samples.input_transform = deepcopy(outcome_model.input_transform)
    gp_samples.outcome_transform = deepcopy(outcome_model.outcome_transform)
    return gp_samples


class BALD(MCAcquisitionFunction):
    r"""Bayesian Active Learning by Disagreement"""

    def __init__(
        self,
        outcome_model: Model,
        pref_model: Model,
        search_space_type: str,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        # sampler and objectives are placeholders and not used
        if sampler is None:
            sampler = PosteriorMeanDummySampler(model=outcome_model)

        if objective is None:
            preference_sampler = SobolQMCNormalSampler(
                num_samples=64, resample=False, collapse_batch_dims=True
            )
            objective = LearnedPrefereceObjective(
                pref_model=pref_model,
                sampler=preference_sampler,
                use_mean=False,
            )

        super().__init__(
            model=outcome_model,
            sampler=sampler,
            objective=objective,
            X_pending=X_pending,
        )

        pref_model.eval()
        self.pref_model = pref_model
        self.search_space_type = search_space_type
        if search_space_type == "rff":
            self.gp_samples = get_rff_sample(outcome_model)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # only work with q = 2
        assert X.shape[-2] == 2

        if self.search_space_type == "rff":
            batch_q_shape = X.shape[:-1]
            X_dim = X.shape[-1]
            Y = self.gp_samples.posterior(X.reshape(-1, X_dim)).mean.reshape(batch_q_shape + (-1,))
        elif self.search_space_type == "f_mean":
            outcome_posterior = self.outcome_model.posterior(X)
            Y = outcome_posterior.mean
        elif self.search_space_type == "y":
            Y = X
        else:
            raise UnsupportedError("Unsupported search_space_type!")

        preference_posterior = self.pref_model(Y)
        preference_mean = preference_posterior.mean
        preference_cov = preference_posterior.covariance_matrix

        mu = preference_mean[..., 0] - preference_mean[..., 1]
        var = (
            2.0
            + preference_cov[..., 0, 0]
            + preference_cov[..., 1, 1]
            - preference_cov[..., 0, 1]
            - preference_cov[..., 1, 0]
        )
        sigma = torch.sqrt(var)

        obj_samples = Normal(0, 1).cdf(Normal(mu, sigma).rsample(torch.Size([2048])))

        posterior_entropies = (
            Bernoulli(Normal(0, 1).cdf(mu / torch.sqrt(var + 1))).entropy().squeeze(-1)
        )

        sample_entropies = Bernoulli(obj_samples).entropy()
        conditional_entropies = sample_entropies.mean(dim=0).squeeze(-1)
        return posterior_entropies - conditional_entropies


class qPreferentialOptimal(MCAcquisitionFunction):
    r"""
    MC EUBO
    (y_1, y_2)^* = argmax_{y_1,y_2 \in Y} E[max{g(y_1), g(y_2)}]
    """

    def __init__(
        self,
        outcome_model: Model,
        pref_model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        r"""q-Preferential Noisy Expected Improvement.

        Args:
            outcome_model (Model): .
            pref_model (Model): .
            sampler (Optional[MCSampler], optional): . Defaults to None.
            objective (Optional[MCAcquisitionObjective], optional): . Defaults to None.
            X_pending (Optional[Tensor], optional): . Defaults to None.
        """

        if sampler is None:
            sampler = PosteriorMeanDummySampler(model=outcome_model)

        if objective is None:
            preference_sampler = SobolQMCNormalSampler(
                num_samples=64, resample=False, collapse_batch_dims=True
            )
            objective = LearnedPrefereceObjective(
                pref_model=pref_model,
                sampler=preference_sampler,
                use_mean=False,
            )

        super().__init__(
            model=outcome_model,
            sampler=sampler,
            objective=objective,
            X_pending=X_pending,
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qNoisyExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Noisy Expected Improvement values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        Y_posterior = self.model.posterior(X)
        Y_samples = self.sampler(Y_posterior)
        obj = self.objective(Y_samples)
        max_util_samples = obj.max(dim=-1).values
        exp_max_util = max_util_samples.mean(dim=0)
        return exp_max_util


class ExpectedUtility(AcquisitionFunction):
    r"""Analytic Prefential Expected Utility, i.e., Analytical EUBO"""

    def __init__(
        self,
        preference_model: Model,
        outcome_model: Model,
        search_space_type: str = "f_mean",
        previous_winner: Optional[Tensor] = None,
    ) -> None:
        r"""Analytic Preferential Expected Utility.

        Args:
            preference_model (Model): .
            outcome_model (Model): .
            search_space_type (str, optional): "f_mean", "rff", or "one_sample". Defaults to "f_mean".
            previous_winner (Optional[Tensor], optional): Tensor representing the previous winner in Y space.
                Defaults to None.
        """
        super().__init__(model=outcome_model)
        self.preference_model = preference_model
        self.outcome_model = outcome_model
        self.register_buffer("previous_winner", previous_winner)
        self.preference_model.eval()  # make sure model is in eval mode
        if self.outcome_model is not None:
            self.outcome_model.eval()
        self.search_space_type = search_space_type
        dtype = preference_model.datapoints.dtype
        device = preference_model.datapoints.device
        self.std_norm = torch.distributions.normal.Normal(
            torch.zeros(1, dtype=dtype, device=device),
            torch.ones(1, dtype=dtype, device=device),
        )
        if search_space_type == "rff":
            self.gp_samples = get_rff_sample(outcome_model)
        elif search_space_type == "one_sample":
            Y_dim = preference_model.datapoints.shape[-1]
            self.w = self.std_norm.rsample((Y_dim,)).squeeze(-1)
        elif search_space_type not in ("y", "f_mean"):
            raise UnsupportedError("Unsupported search space!")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate PreferentialOneStepLookahead on the candidate set X.
        Args:
            X: A `batch_shape x q x d`-dim Tensor, where `q = 2` if `previous_winner` is
            not `None`, and `q = 1` otherwise.
        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        assert (X.shape[-2] == 2) or ((X.shape[-2] == 1) and (self.previous_winner is not None))

        if self.search_space_type == "rff":
            batch_q_shape = X.shape[:-1]
            X_dim = X.shape[-1]
            Y = self.gp_samples.posterior(X.reshape(-1, X_dim)).mean.reshape(batch_q_shape + (-1,))
        elif self.search_space_type == "f_mean":
            outcome_posterior = self.outcome_model.posterior(X)
            Y = outcome_posterior.mean
        elif self.search_space_type == "one_sample":
            post = self.outcome_model.posterior(X)
            Y = post.mean + post.variance.sqrt() * self.w
        elif self.search_space_type == "y":
            Y = X
        else:
            raise UnsupportedError("Unsupported search_space_type!")

        if self.previous_winner is not None:
            Y = torch.cat([Y, match_batch_shape(self.previous_winner, Y)], dim=-2)
        preference_posterior = self.preference_model(Y)
        preference_mean = preference_posterior.mean
        preference_cov = preference_posterior.covariance_matrix
        delta = preference_mean[..., 0] - preference_mean[..., 1]
        sigma = torch.sqrt(
            preference_cov[..., 0, 0]
            + preference_cov[..., 1, 1]
            - preference_cov[..., 0, 1]
            - preference_cov[..., 1, 0]
        )
        u = delta / sigma

        ucdf = self.std_norm.cdf(u)
        updf = torch.exp(self.std_norm.log_prob(u))
        acqf_val = sigma * (updf + u * ucdf)
        if self.previous_winner is None:
            acqf_val += preference_mean[..., 1]
        return acqf_val
