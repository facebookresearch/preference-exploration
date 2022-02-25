import random
import time
import warnings
from copy import deepcopy
from itertools import permutations

import numpy as np
import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.utils.gp_sampling import get_gp_samples
from scipy.optimize import minimize

from acquisition_functions import ExpectedUtility
from constants import *  # noqa: F403, F401
from sim_helpers import fit_pref_model, organize_comparisons

warnings.filterwarnings("ignore", message="Could not update `train_inputs` with transformed inputs")


def pref2rff(pref_model, n_samples):
    # assume pref_model on cpu
    pref_model = pref_model.eval().double()
    # force the model to infer utility
    pref_model.posterior(pref_model.datapoints)
    modified_pref_model = deepcopy(pref_model)

    class LikelihoodForRFF:
        noise = torch.tensor(1.0).double()

    modified_pref_model.likelihood = LikelihoodForRFF()
    modified_pref_model.train_targets = pref_model.utility
    modified_pref_model.input_transforms = None

    gp_samples = get_gp_samples(
        model=modified_pref_model,
        num_outputs=1,
        n_samples=n_samples,
        num_rff_features=500,
    )
    gp_samples.input_transform = deepcopy(pref_model.input_transform)

    # gp_samples = gp_samples.to(device=device)

    return gp_samples


def get_pbo_pe_comparisons(
    outcome_X,
    train_comps,
    problem,
    utils,
    init_round,
    total_training_round,
    comp_noise_type,
    comp_noise,
    pe_strategy,
):
    """
    Generate TS-based comparisons on previously observed points

    Args:
        outcome_X ([type]): [description]
        train_comps ([type]): [description]
        problem ([type]): [description]
        utils ([type]): [description]
        init_round ([type]): [description]
        total_training_round ([type]): [description]
        comp_noise_type ([type]): [description]
        comp_noise ([type]): [description]
        pe_strategy (str): being either "random", "ts" or "eubo"

    Returns:
        [type]: [description]
    """

    all_pairs = torch.combinations(torch.tensor(range(outcome_X.shape[-2])), r=2).to(train_comps)

    for i in range(total_training_round):
        pbo_pe_start_time = time.time()
        if (
            (pe_strategy != "random")
            and (train_comps is not None)
            and (train_comps.shape[-2] >= init_round)
        ):
            pbo_pref_model = fit_pref_model(
                outcome_X,
                train_comps,
                kernel="default",
                transform_input=True,
                Y_bounds=problem.bounds,
            )

        if (
            (pe_strategy == "random")
            or (train_comps is None)
            or (train_comps.shape[-2] < init_round)
        ):
            cand_comps = all_pairs[
                torch.randint(high=all_pairs.shape[-2], size=(1,)),
            ]
        elif pe_strategy == "ts":
            cand_comps = None
            # use TS to draw comparisons
            comp1 = pbo_pref_model.posterior(outcome_X).sample().argmax(dim=-2)
            # exclude the first sample
            sample2 = pbo_pref_model.posterior(outcome_X).sample()
            sample2[:, comp1.squeeze(), :] = -float("Inf")
            comp2 = sample2.argmax(dim=-2)
            # Create candidate comparisons
            cand_comps = torch.cat((comp1, comp2), dim=-1)
        elif pe_strategy == "eubo":
            eubo_acqf = ExpectedUtility(
                preference_model=pbo_pref_model,
                outcome_model=None,
                previous_winner=None,
                search_space_type="y",
            )
            cand_comps = None
            max_eubo_val = -np.inf
            for j in range(all_pairs.shape[-2]):
                X_pair = outcome_X[all_pairs[j, :]]
                eubo_val = eubo_acqf(X_pair).item()

                if eubo_val > max_eubo_val:
                    max_eubo_val = eubo_val
                    cand_comps = all_pairs[[j], :]
        else:
            raise ValueError("Unsupported PE strategy for PBO")
        cand_comps = organize_comparisons(utils, cand_comps, comp_noise_type, comp_noise)
        pbo_pe_time = time.time() - pbo_pe_start_time

        train_comps = cand_comps if train_comps is None else torch.cat((train_comps, cand_comps))
        print(
            f"PBO with PE strategy {pe_strategy} gen time: {pbo_pe_time:.2f}s, train_comps shape: {train_comps.shape}"
        )

    return train_comps


def gen_pbo_candidates(outcome_X, train_comps, q, problem, pbo_gen_method):
    """generate pbo candidates

    Args:
        outcome_X (_type_): _description_
        train_comps (_type_): _description_
        q (_type_): _description_
        problem (_type_): _description_
        pbo_gen_method (_type_): _description_
    """
    if pbo_gen_method == "ts":
        problem_cpu = deepcopy(problem).cpu()
        pref_model = fit_pref_model(
            outcome_X, train_comps, kernel="default", transform_input=True, Y_bounds=problem.bounds
        )

        outcome_cand_X = []
        for _ in range(q):
            gp_samples = pref2rff(pref_model.cpu(), n_samples=1)

            acqf = PosteriorMean(gp_samples)
            single_outcome_cand_X, _ = optimize_acqf(
                acqf,
                bounds=problem_cpu.bounds,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                options={"batch_limit": 1},
            )

            outcome_cand_X.append(single_outcome_cand_X)

        outcome_cand_X = torch.cat(outcome_cand_X).to(outcome_X)
    elif pbo_gen_method == "ei":
        pref_model = fit_pref_model(
            outcome_X, train_comps, kernel="default", transform_input=True, Y_bounds=problem.bounds
        )

        # to fill in utility values
        pref_model.posterior(pref_model.datapoints)

        acqf = qExpectedImprovement(model=pref_model, best_f=pref_model.utility.max().item())

        outcome_cand_X, _ = optimize_acqf(
            acqf,
            bounds=problem.bounds,
            q=q,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            sequential=True,
        )
    else:
        raise ValueError("Unsupported gen_method for PBO")

    return outcome_cand_X
