import itertools
import random
import time
from copy import deepcopy

import cma
import torch
from botorch.acquisition import GenericMCObjective
from botorch.acquisition.monte_carlo import (qNoisyExpectedImprovement,
                                             qSimpleRegret)
from botorch.acquisition.utils import prune_inferior_points
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.pairwise_gp import (PairwiseGP,
                                        PairwiseLaplaceMarginalLogLikelihood)
from botorch.models.transforms.input import (ChainedInputTransform, Normalize,
                                             Warp)
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.scalarization import \
    get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from gpytorch.kernels import (AdditiveStructureKernel, LinearKernel,
                              PolynomialKernel, ScaleKernel)
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.mlls.exact_marginal_log_likelihood import \
    ExactMarginalLogLikelihood
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch.priors.torch_priors import LogNormalPrior
from scipy.stats import kendalltau

from acquisition_functions import BALD, ExpectedUtility
from constants import *
from helper_classes import LearnedPrefereceObjective, PosteriorMeanDummySampler
from test_functions import gen_rand_points, gen_rand_X, problem_setup


def fit_outcome_model(X, Y, X_bounds):
    # fit outcome model
    input_tf = Normalize(d=X.shape[-1], bounds=X_bounds)
    outcome_model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        outcome_transform=Standardize(m=Y.shape[-1]),
        input_transform=input_tf,
    )
    mll = ExactMarginalLogLikelihood(outcome_model.likelihood, outcome_model)
    fit_gpytorch_model(mll)
    return outcome_model.to(device=X.device, dtype=X.dtype)


def fit_pref_model(Y, comps, kernel, transform_input=True, Y_bounds=None, jitter=1e-4):
    """Preference model fitting helper function"""
    Y_dim = Y.shape[-1]
    if Y_bounds is None or not transform_input:
        chained_transform = None
    else:
        normalize_tf = Normalize(d=Y_dim, bounds=Y_bounds)
        warp_tf = Warp(
            indices=list(range(Y_dim)),
            # use a prior with median at 1.
            # when a=1 and b=1, the Kumaraswamy CDF is the identity function
            concentration1_prior=LogNormalPrior(0.0, 0.75 ** 0.5),
            concentration0_prior=LogNormalPrior(0.0, 0.75 ** 0.5),
        )
        chained_transform = ChainedInputTransform(normalize_tf=normalize_tf, warp_tf=warp_tf)

    if kernel == "default":
        model = PairwiseGP(
            Y.double().cpu(),
            comps.double().cpu(),
            jitter=jitter,
            input_transform=chained_transform,
        )
    elif kernel == "linear":
        covar_module = ScaleKernel(
            LinearKernel(num_dimensions=Y.shape[-1]),
            outputscale_prior=SmoothedBoxPrior(a=1, b=4),
        )
        model = PairwiseGP(
            Y.double().cpu(),
            comps.double().cpu(),
            covar_module=covar_module,
            jitter=jitter,
            input_transform=chained_transform,
        )
    elif kernel == "additive":
        covar_module = AdditiveStructureKernel(base_kernel=RBFKernel(), num_dims=Y.shape[1])
        model = PairwiseGP(
            Y.double().cpu(),
            comps.double().cpu(),
            covar_module=covar_module,
            jitter=jitter,
            input_transform=chained_transform,
        )
    elif kernel == "polynomial":
        covar_module = ScaleKernel(
            PolynomialKernel(power=2),
            outputscale_prior=SmoothedBoxPrior(a=1, b=2),
        )
        model = PairwiseGP(
            Y.double().cpu(),
            comps.double().cpu(),
            covar_module=covar_module,
            jitter=jitter,
            input_transform=chained_transform,
        )
    else:
        raise RuntimeError("Unsupported kernel")
    mll = PairwiseLaplaceMarginalLogLikelihood(model)
    fit_gpytorch_model(mll)
    model = model.to(device=Y.device, dtype=Y.dtype)

    return model


def inject_comp_error(comp, util_diff, comp_noise_type, comp_noise):
    std_norm = torch.distributions.normal.Normal(
        torch.zeros(1, dtype=util_diff.dtype, device=util_diff.device),
        torch.ones(1, dtype=util_diff.dtype, device=util_diff.device),
    )

    if comp_noise_type == "constant":
        comp_error_p = comp_noise
    elif comp_noise_type == "probit":
        comp_error_p = 1 - std_norm.cdf(util_diff / comp_noise)
    else:
        raise UnsupportedError(f"Unsupported comp_noise_type: {comp_noise_type}")

    # with comp_error_p probability to make a comparison mistake
    flip_rand = torch.rand(util_diff.shape).to(util_diff)
    to_flip = flip_rand < comp_error_p
    flipped_comp = comp.clone()
    if len(flipped_comp.shape) > 1:
        assert (util_diff >= 0).all()
        # flip tensor
        flipped_comp[to_flip, 0], flipped_comp[to_flip, 1] = comp[to_flip, 1], comp[to_flip, 0]
    else:
        assert util_diff > 0
        # flip a single pair
        if to_flip:
            flipped_comp[[0, 1]] = flipped_comp[[1, 0]]
    return flipped_comp


def organize_comparisons(utils, comps, comp_noise_type, comp_noise):
    """
    Given utility and comparisons in arbitrary orders,
    re-order comparisons such that comparisons are in correct orders
    with comparison noise injected

    Args:
        utils ([type]): [description]
        comps ([type]): [description]
        comp_noise_type ([type]): [description]
        comp_noise ([type]): [description]

    Returns:
        [type]: [description]
    """
    comps = deepcopy(comps)
    pair_utils = utils[comps]
    is_incorrect = pair_utils[..., 0] < pair_utils[..., 1]
    comps[is_incorrect, 0], comps[is_incorrect, 1] = (
        comps[is_incorrect, 1],
        comps[is_incorrect, 0],
    )

    # inject comparison error
    util_diff = utils[comps]
    util_diff = util_diff[..., 0] - util_diff[..., 1]
    comps = inject_comp_error(
        comps, util_diff, comp_noise_type=comp_noise_type, comp_noise=comp_noise
    )
    comps = comps.to(device=utils.device)
    return comps


def gen_comps(utility, comp_noise_type, comp_noise):
    """Create pairwise comparisons"""
    cpu_util = utility.cpu()

    comp_pairs = []
    for i in range(cpu_util.shape[0] // 2):
        i1 = i * 2
        i2 = i * 2 + 1
        if cpu_util[i1] > cpu_util[i2]:
            new_comp = [i1, i2]
            util_diff = cpu_util[i1] - cpu_util[i2]
        else:
            new_comp = [i2, i1]
            util_diff = cpu_util[i2] - cpu_util[i1]

        new_comp = torch.tensor(new_comp, device=utility.device, dtype=torch.long)
        new_comp = inject_comp_error(new_comp, util_diff, comp_noise_type, comp_noise)
        comp_pairs.append(new_comp)

    comp_pairs = torch.stack(comp_pairs)

    return comp_pairs


def gen_initial_real_data(n, problem, get_util):
    # generate (noisy) ground truth data
    X = gen_rand_X(n, problem)
    Y = problem(X)
    util = get_util(Y)
    comps = gen_comps(util, comp_noise_type="constant", comp_noise=0)

    return X, Y, util, comps


def gen_exp_comps(X, model, get_util, comp_noise_type, comp_noise, sample_outcome):
    # generate posterior dras and make simulated comparisons based on that
    if sample_outcome:
        cand_Y = model.posterior(X).sample().squeeze(0)
    else:
        cand_Y = model.posterior(X).mean.clone().detach()
    cand_util = get_util(cand_Y)
    cand_comps = gen_comps(cand_util, comp_noise_type, comp_noise)
    return cand_Y, cand_util, cand_comps


def gen_uncorrelated_candidates(q, Y_bounds, get_util, comp_noise_type, comp_noise):
    # randomly selected points in Y space
    cand_X = torch.tensor([]).to(Y_bounds)
    cand_Y = gen_rand_points(q, Y_bounds.shape[-1], Y_bounds)
    cand_util = get_util(cand_Y)
    cand_comps = gen_comps(cand_util, comp_noise_type, comp_noise)

    return cand_X, cand_Y, cand_util, cand_comps


def gen_random_candidates(model, q, problem, get_util, comp_noise_type, comp_noise, sample_outcome):
    # generate training data
    cand_X = gen_rand_X(q, problem=problem)
    cand_Y, cand_util, cand_comps = gen_exp_comps(
        cand_X, model, get_util, comp_noise_type, comp_noise, sample_outcome
    )
    return cand_X, cand_Y, cand_util, cand_comps


def gen_observed_candidates(
    outcome_X, outcome_Y, selected_pairs, get_util, comp_noise_type, comp_noise
):
    # randomly selected observed points

    # all combination of index pairs
    all_combo = list(itertools.combinations(range(outcome_Y.shape[0]), 2))
    new_comp = all_combo[random.randrange(len(all_combo))]

    # put the new pair in right order
    new_util = get_util(outcome_Y[new_comp, :])
    if new_util[1] > new_util[0]:
        new_comp = (new_comp[1], new_comp[0])
        util_diff = new_util[1] - new_util[0]
    else:
        util_diff = new_util[0] - new_util[1]

    new_comp = inject_comp_error(new_comp, util_diff, comp_noise_type, comp_noise)

    selected_pairs.append(new_comp)
    unique_ids = torch.tensor(selected_pairs).flatten().unique().tolist()
    id_map = dict(zip(unique_ids, range(len(unique_ids))))

    # construct all candidate values (instead of only new ones)
    all_cand_X = outcome_X[unique_ids, :]
    all_cand_Y = outcome_Y[unique_ids, :]
    all_cand_util = get_util(all_cand_Y)
    all_cand_comps = torch.tensor([(id_map[p1], id_map[p2]) for p1, p2 in selected_pairs])

    return all_cand_X, all_cand_Y, all_cand_util, all_cand_comps, selected_pairs


def gen_parego_candidates(
    model, X, Y, q, problem, get_util, comp_noise_type, comp_noise, sample_outcome, gen_method
):
    cand_X = []
    for _ in range(q):
        weights = sample_simplex(problem.num_objectives).squeeze().to(Y)
        objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=Y))

        if gen_method == "ts":
            n_sample = 1024

            rand_X = gen_rand_X(n_sample, problem)
            outcome_post = model.posterior(rand_X).sample().squeeze(0)
            post_util = objective(outcome_post)
            cand_X.append(rand_X[torch.argmax(post_util), :])
        else:
            try:
                sampler = SobolQMCNormalSampler(num_samples=NUM_PAREGO_SAMPLES)
                if gen_method == "qnei":
                    acq_func = qNoisyExpectedImprovement(
                        model=model,
                        X_baseline=X,
                        sampler=sampler,
                        objective=objective,
                        prune_baseline=True,
                    )
                else:
                    raise RuntimeError

                # optimize
                # generate 1 candidate at a time, repeat q times
                single_cand_X, _ = optimize_acqf(
                    acq_function=acq_func,
                    q=1,
                    bounds=problem.bounds,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                    options={"batch_limit": BATCH_LIMIT, "ftol": FTOL},
                )
                cand_X.append(single_cand_X.squeeze(0))
            except UnsupportedError:
                sampler = IIDNormalSampler(num_samples=NUM_PAREGO_SAMPLES)
                if gen_method == "qnei":
                    acq_func = qNoisyExpectedImprovement(
                        model=model,
                        X_baseline=X,
                        sampler=sampler,
                        objective=objective,
                        prune_baseline=True,
                    )
                else:
                    raise RuntimeError

                # optimize
                # generate 1 candidate at a time, repeat q times
                single_cand_X, _ = optimize_acqf(
                    acq_function=acq_func,
                    q=1,
                    bounds=problem.bounds,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                    options={"batch_limit": BATCH_LIMIT, "ftol": FTOL},
                )
                cand_X.append(single_cand_X.squeeze(0))

    cand_X = torch.stack(cand_X)
    # "observe" new values from outcome model
    cand_Y, cand_util, cand_comps = gen_exp_comps(
        cand_X, model, get_util, comp_noise_type, comp_noise, sample_outcome
    )

    return cand_X, cand_Y, cand_util, cand_comps


def gen_true_util_data(model, X, Y, q, problem, get_util, comp_noise_type, comp_noise, gen_method):
    sampler = SobolQMCNormalSampler(num_samples=NUM_TRUE_UTIL_SAMPLES)
    true_obj = GenericMCObjective(get_util)

    if gen_method == "ts":
        cand_X = []
        n_sample = 1024

        for i in range(q):
            rand_X = gen_rand_X(n_sample, problem)
            outcome_post = model.posterior(rand_X).sample().squeeze(0)
            post_util = true_obj(outcome_post)
            cand_X.append(rand_X[torch.argmax(post_util), :])

        cand_X = torch.stack(cand_X)

    else:
        if gen_method == "qnei":
            acq_func = qNoisyExpectedImprovement(
                model=model,
                X_baseline=X[:1],
                sampler=sampler,
                objective=true_obj,
                prune_baseline=False,
            )
        else:
            raise RuntimeError
        cand_X, _ = optimize_acqf(
            acq_function=acq_func,
            q=q,
            bounds=problem.bounds,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": BATCH_LIMIT, "ftol": FTOL},
        )

    # "observe" new values from outcome model
    cand_Y, cand_util, cand_comps = gen_exp_comps(
        cand_X,
        model,
        get_util,
        comp_noise_type,
        comp_noise,
        sample_outcome=False,
    )
    return cand_X, cand_Y, cand_util, cand_comps


def get_pref_acqf(
    outcome_model,
    pref_model,
    X_baseline,
    problem,
    sampler_constructor,
    gen_method,
    **kwargs,
):
    prune_pref_sample_num = kwargs.get("prune_pref_sample_num", 64)
    prune_outcome_sample_num = kwargs.get("prune_outcome_sample_num", 64)
    pref_mean = kwargs.get("pref_mean", False)
    pref_sample_num = kwargs.get("pref_sample_num", NUM_LEARN_PREF_SAMPLES_UNEIPM)
    outcome_mean = kwargs.get("outcome_mean", True)
    outcome_sample_num = kwargs.get("outcome_sample_num", 1)
    device = kwargs.get("device", torch.device("cpu"))
    dtype = kwargs.get("dtype", torch.float)

    print(f"Inside pref_mean: {pref_mean}, {pref_sample_num}, {outcome_mean}, {outcome_sample_num}")

    prune_obj = LearnedPrefereceObjective(
        pref_model=pref_model,
        sampler=sampler_constructor(num_samples=prune_pref_sample_num),
        use_mean=False,
    ).to(device=device, dtype=dtype)

    if outcome_mean:
        prune_sampler = PosteriorMeanDummySampler(model=outcome_model)
    else:
        prune_sampler = sampler_constructor(num_samples=prune_outcome_sample_num)

    pruned_train_X = prune_inferior_points(
        model=outcome_model,
        X=X_baseline,
        objective=prune_obj,
        sampler=prune_sampler,
    )

    pref_obj = LearnedPrefereceObjective(
        pref_model=pref_model,
        sampler=sampler_constructor(num_samples=pref_sample_num),
        use_mean=pref_mean,
    )
    if outcome_mean:
        outcome_sampler = PosteriorMeanDummySampler(model=outcome_model)
    else:
        outcome_sampler = sampler_constructor(num_samples=outcome_sample_num)

    if gen_method == "qnei":
        acq_func = qNoisyExpectedImprovement(
            model=outcome_model,
            X_baseline=pruned_train_X,
            sampler=outcome_sampler,
            objective=pref_obj,
            prune_baseline=False,
        )
    else:
        raise RuntimeError(f"unsupported qnei gen method {gen_method}")

    return acq_func


def gen_bald_candidates(
    outcome_model, pref_model, problem, gen_method, Y_bounds, search_space_type, **kwargs
):
    q = kwargs.get("q", 1)
    num_restarts = kwargs.get("num_restarts", NUM_RESTARTS)
    raw_samples = kwargs.get("raw_samples", RAW_SAMPLES)
    batch_limit = kwargs.get("batch_limit", BATCH_LIMIT)
    sequential = kwargs.get("sequential", False)

    print(f"BALD q={q}, search_space_type={search_space_type}")
    if gen_method == "ts":
        raise RuntimeError("Can't do TS!")

    if search_space_type == "y":
        bounds = Y_bounds
    else:
        bounds = problem.bounds

    acqf = BALD(
        outcome_model=outcome_model, pref_model=pref_model, search_space_type=search_space_type
    )

    cand_X, acqf_val = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": batch_limit, "ftol": FTOL},
        sequential=sequential,
    )

    if search_space_type == "rff":
        cand_Y = acqf.gp_samples.posterior(cand_X).mean.squeeze(0).clone().detach()
    elif search_space_type == "f_mean":
        cand_Y = outcome_model.posterior(cand_X).mean.clone().detach()
    elif search_space_type == "y":
        cand_Y = cand_X
        # create empty tensor so that it won't trigger issues when we append it to train_X
        cand_X = torch.empty(0).to(cand_Y)
    else:
        raise UnsupportedError("Unsupported search_space_type!")

    return cand_X, cand_Y, acqf_val


def gen_expected_util_candidates(
    outcome_model, pref_model, problem, previous_winner, search_space_type, **kwargs
):
    """Analytical EUBO"""
    q = 2 if previous_winner is None else 1
    num_restarts = kwargs.get("num_restarts", NUM_RESTARTS)
    raw_samples = kwargs.get("raw_samples", RAW_SAMPLES)
    batch_limit = kwargs.get("batch_limit", BATCH_LIMIT)
    sequential = kwargs.get("sequential", False)
    Y_bounds = kwargs.get("Y_bounds", False)
    return_acqf = kwargs.get("return_acqf", False)

    if search_space_type == "y":
        bounds = Y_bounds
    else:
        bounds = problem.bounds

    acqf = ExpectedUtility(
        preference_model=pref_model,
        outcome_model=outcome_model,
        previous_winner=previous_winner,
        search_space_type=search_space_type,
    )

    cand_X, acqf_val = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": batch_limit, "ftol": FTOL},
        sequential=sequential,
    )

    if search_space_type == "rff":
        cand_Y = acqf.gp_samples.posterior(cand_X).mean.squeeze(0).clone().detach()
    elif search_space_type == "f_mean":
        cand_Y = outcome_model.posterior(cand_X).mean.clone().detach()
    elif search_space_type == "one_sample":
        post = outcome_model.posterior(cand_X)
        cand_Y = (post.mean + post.variance.sqrt() * acqf.w).clone().detach()
    elif search_space_type == "y":
        cand_Y = cand_X
        # create empty tensor so that it won't trigger issues when we append it to train_X
        cand_X = torch.empty(0).to(cand_Y)
    else:
        raise UnsupportedError("Unsupported search_space_type!")
    if return_acqf:
        return cand_X, cand_Y, acqf_val, acqf
    else:
        return cand_X, cand_Y, acqf_val


def gen_pref_candidates(outcome_model, pref_model, X_baseline, problem, gen_method, **kwargs):
    q = kwargs.get("q", 1)
    num_restarts = kwargs.get("num_restarts", NUM_RESTARTS)
    raw_samples = kwargs.get("raw_samples", RAW_SAMPLES)
    batch_limit = kwargs.get("batch_limit", BATCH_LIMIT)
    sequential = kwargs.get("sequential", False)

    try:
        sampler_constructor = SobolQMCNormalSampler
        acqf = get_pref_acqf(
            outcome_model,
            pref_model,
            X_baseline,
            problem,
            gen_method=gen_method,
            sampler_constructor=sampler_constructor,
            **kwargs,
        )
        cand_X, acqf_val = optimize_acqf(
            acq_function=acqf,
            bounds=problem.bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": batch_limit, "ftol": FTOL},
            sequential=sequential,
        )
    except UnsupportedError:
        "Switch to IID normal sampler if sobol fails"
        sampler_constructor = IIDNormalSampler
        acqf = get_pref_acqf(
            outcome_model,
            pref_model,
            X_baseline,
            problem,
            gen_method=gen_method,
            sampler_constructor=sampler_constructor,
            **kwargs,
        )
        cand_X, acqf_val = optimize_acqf(
            acq_function=acqf,
            bounds=problem.bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": batch_limit, "ftol": FTOL},
            sequential=sequential,
        )

    return cand_X, acqf_val


def gen_pref_candidates_eval(
    outcome_model, pref_model, X_baseline, problem, q, gen_method, tkwargs
):
    return gen_pref_candidates(
        outcome_model=outcome_model,
        pref_model=pref_model,
        X_baseline=X_baseline,
        problem=problem,
        gen_method=gen_method,
        q=q,
        num_restarts=NUM_RESTARTS,
        batch_limit=BATCH_LIMIT,
        raw_samples=RAW_SAMPLES,
        sequential=True,
        pref_sample_num=NUM_EVAL_PREF_SAMPLES,
        pref_mean=False,
        outcome_sample_num=NUM_EVAL_OUTCOME_SAMPLES,
        outcome_mean=False,
        **tkwargs,
    )


def gen_post_mean(outcome_model, pref_model, problem, **kwargs):
    pref_sample_num = kwargs.get("pref_sample_num", 64)
    outcome_sample_num = kwargs.get("outcome_sample_num", 64)
    num_restarts = kwargs.get("num_restarts", NUM_RESTARTS)
    batch_limit = kwargs.get("batch_limit", BATCH_LIMIT)
    raw_samples = kwargs.get("raw_samples", RAW_SAMPLES)
    use_mean = kwargs.get("use_mean", False)

    pref_obj = LearnedPrefereceObjective(
        pref_model=pref_model,
        sampler=SobolQMCNormalSampler(num_samples=pref_sample_num),
        use_mean=use_mean,
    )
    outcome_sampler = SobolQMCNormalSampler(num_samples=outcome_sample_num)
    post_mean = qSimpleRegret(
        outcome_model,
        sampler=outcome_sampler,
        objective=pref_obj,
    )
    cand_X, _ = optimize_acqf(
        acq_function=post_mean,
        bounds=problem.bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": batch_limit, "ftol": FTOL},
    )
    return cand_X


def gen_learn_candidates(
    q,
    problem,
    get_util,
    Y_bounds,
    learn_strategy,
    outcome_model,
    pref_model,
    gen_method,
    X_baseline,
    outcome_Y,
    train_Y,
    comp_noise_type,
    comp_noise,
    sample_outcome,
    previous_winner_idx,
    kernel,
    **kwargs,
):

    cand_X = None
    cand_Y = None

    # import pdb; pdb.set_trace()
    for _ in range(2):
        try:
            if learn_strategy == "qnei":
                # uNEI-PM
                cand_X, _ = gen_pref_candidates(
                    outcome_model=outcome_model,
                    pref_model=pref_model,
                    X_baseline=X_baseline,
                    problem=problem,
                    gen_method=gen_method,
                    q=q,
                    **kwargs,
                )
            elif learn_strategy in ("bald_correct", "bald_yspace", "bald_rff"):
                if learn_strategy == "bald_correct":
                    search_space_type = "f_mean"
                elif learn_strategy == "bald_yspace":
                    search_space_type = "y"
                elif learn_strategy == "bald_rff":
                    search_space_type = "rff"
                else:
                    raise UnsupportedError("Uknown BALD search_space_type!")

                cand_X, cand_Y, _ = gen_bald_candidates(
                    outcome_model=outcome_model,
                    pref_model=pref_model,
                    problem=problem,
                    gen_method=gen_method,
                    q=q,
                    Y_bounds=Y_bounds,
                    search_space_type=search_space_type,
                    **kwargs,
                )
            elif learn_strategy == "eubo_rff":
                # EUBO-PS
                cand_X, cand_Y, _ = gen_expected_util_candidates(
                    outcome_model=outcome_model,
                    pref_model=pref_model,
                    problem=problem,
                    previous_winner=None,
                    search_space_type="rff",
                )
            elif learn_strategy == "eubo_one_sample":
                # EUBO-OPS
                cand_X, cand_Y, _ = gen_expected_util_candidates(
                    outcome_model=outcome_model,
                    pref_model=pref_model,
                    problem=problem,
                    previous_winner=None,
                    search_space_type="one_sample",
                )
            elif learn_strategy == "eubo_y":
                # EUBO-PS
                cand_X, cand_Y, _ = gen_expected_util_candidates(
                    outcome_model=outcome_model,
                    pref_model=pref_model,
                    problem=problem,
                    previous_winner=None,
                    search_space_type="y",
                    Y_bounds=Y_bounds,
                )
            elif learn_strategy in ("random", "random_ps"):
                # Surrogate(-ps) random
                if learn_strategy == "random_ps":
                    sample_outcome = True
                cand_X, cand_Y, _, _ = gen_random_candidates(
                    model=outcome_model,
                    q=q,
                    problem=problem,
                    get_util=get_util,
                    comp_noise_type=comp_noise_type,
                    comp_noise=comp_noise,
                    sample_outcome=sample_outcome,
                )
            elif learn_strategy == "uncorrelated":
                # Uniform random
                cand_X, cand_Y, _, _ = gen_uncorrelated_candidates(
                    q=q,
                    Y_bounds=Y_bounds,
                    get_util=get_util,
                    comp_noise_type=comp_noise_type,
                    comp_noise=comp_noise,
                )
            else:
                raise RuntimeError("Unsupported learning strategy")
        except Exception as e:
            print(e)
            print("Encounter exceptions... try again...")
            continue
        break

    if cand_Y is None:
        if sample_outcome:
            cand_Y = outcome_model.posterior(cand_X).sample().squeeze(0).clone().detach()
        else:
            cand_Y = outcome_model.posterior(cand_X).mean.clone().detach()

    return cand_X, cand_Y


def run_one_round_sim(
    total_training_round,
    init_round,
    problem_str,
    noisy,
    comp_noise_type,
    comp_noise,
    outcome_model,
    outcome_X,
    outcome_Y,
    train_X,
    train_Y,
    train_comps,
    init_strategy,
    learn_strategy,
    gen_method,
    keep_winner_prob,
    sample_outcome,
    kernel,
    check_post_mean,
    check_post_mean_every_k,
    tkwargs,
    selected_pairs,  # for "observed" init/learn strategy only, set to [] by default
):

    (
        X_dim,
        Y_dim,
        problem,
        util_type,
        get_util,
        Y_bounds,
        probit_noise,
    ) = problem_setup(problem_str, noisy=noisy, **tkwargs)

    pref_model = None
    last_winner_idx = None
    post_mean_X = []
    post_mean_idx = []
    run_times = []
    acq_run_times = []

    # if started with previous train_Y, initialize the pref model
    if train_Y is not None:
        pref_model = fit_pref_model(
            train_Y, train_comps, kernel=kernel, transform_input=True, Y_bounds=Y_bounds
        )

    for i in range(total_training_round):
        start_time = time.time()
        if i < init_round or keep_winner_prob is None or learn_strategy == "observed":
            # Init phase
            current_strategy = init_strategy
            # using initialization strategy
            if init_strategy in ("random", "random_ps"):
                if init_strategy == "random_ps":
                    sample_outcome = True
                (pref_init_X, pref_init_Y, _, pref_init_comps,) = gen_random_candidates(
                    model=outcome_model,
                    q=2,
                    problem=problem,
                    get_util=get_util,
                    comp_noise_type=comp_noise_type,
                    comp_noise=comp_noise,
                    sample_outcome=sample_outcome,
                )
            elif init_strategy == "parego":
                if train_X is None:
                    X_baseline = outcome_X
                else:
                    X_baseline = torch.cat((train_X, outcome_X)).to(**tkwargs)
                (pref_init_X, pref_init_Y, _, pref_init_comps,) = gen_parego_candidates(
                    model=outcome_model,
                    X=X_baseline,
                    Y=outcome_Y,
                    q=2,
                    problem=problem,
                    get_util=get_util,
                    comp_noise_type=comp_noise_type,
                    comp_noise=comp_noise,
                    sample_outcome=sample_outcome,
                    gen_method=gen_method,
                )
            elif init_strategy == "uncorrelated":
                # not using sample_outcome
                (pref_init_X, pref_init_Y, _, pref_init_comps,) = gen_uncorrelated_candidates(
                    q=2,
                    Y_bounds=Y_bounds,
                    get_util=get_util,
                    comp_noise_type=comp_noise_type,
                    comp_noise=comp_noise,
                )
            elif init_strategy == "observed":
                # not using sample_outcome
                (
                    pref_init_X,
                    pref_init_Y,
                    _,
                    pref_init_comps,
                    selected_pairs,
                ) = gen_observed_candidates(
                    outcome_X, outcome_Y, selected_pairs, get_util, comp_noise_type, comp_noise
                )
                # manually set train_X to be None so that we can update the whole training data
                train_X = None
            else:
                raise RuntimeError

            if train_X is None:
                train_X = pref_init_X
                train_Y = pref_init_Y
                train_comps = pref_init_comps
            else:
                comps_shifted = pref_init_comps + train_Y.shape[0]
                train_X = torch.cat((train_X, pref_init_X), dim=0)
                train_Y = torch.cat((train_Y, pref_init_Y), dim=0)
                train_comps = torch.cat((train_comps, comps_shifted), dim=0)
        else:
            # Learning phase
            current_strategy = learn_strategy
            X_baseline = torch.cat((train_X, outcome_X)).to(**tkwargs)
            if last_winner_idx is not None and random.random() < keep_winner_prob:
                keep_winner = True
                q = 1
            else:
                keep_winner = False
                q = 2

            # generate candidate(s)
            cand_X, cand_Y = gen_learn_candidates(
                q=q,
                problem=problem,
                get_util=get_util,
                Y_bounds=Y_bounds,
                learn_strategy=learn_strategy,
                outcome_model=outcome_model,
                pref_model=pref_model,
                gen_method=gen_method,
                X_baseline=X_baseline,
                outcome_Y=outcome_Y,
                train_Y=train_Y,
                comp_noise_type=comp_noise_type,
                comp_noise=comp_noise,
                sample_outcome=sample_outcome,
                previous_winner_idx=last_winner_idx if keep_winner else None,
                kernel=kernel,
                **tkwargs,
            )

            if keep_winner:
                new_util = get_util(cand_Y)[0]
                last_winner_util = get_util(train_Y[[last_winner_idx], :])[0]
                new_idx = train_Y.shape[0]

                if new_util > last_winner_util:
                    new_comp = [new_idx, last_winner_idx]
                    util_diff = new_util - last_winner_util
                else:
                    new_comp = [last_winner_idx, new_idx]
                    util_diff = last_winner_util - new_util

                new_comp = torch.tensor(new_comp, device=train_Y.device, dtype=torch.long)
                cand_comps = inject_comp_error(new_comp, util_diff, comp_noise_type, comp_noise)
                cand_comps = cand_comps.unsqueeze(0)
            else:
                cand_util = get_util(cand_Y)
                assert cand_util.shape[0] == 2
                cand_comps = gen_comps(cand_util, comp_noise_type, comp_noise) + train_Y.shape[0]

            train_X = torch.cat((train_X, cand_X)).to(**tkwargs)
            train_Y = torch.cat((train_Y, cand_Y)).to(**tkwargs)
            train_comps = torch.cat((train_comps, cand_comps))
            last_winner_idx = train_comps[-1, 0]

        acq_run_time = time.time() - start_time
        acq_run_times.append(acq_run_time)

        pref_model = fit_pref_model(
            train_Y, train_comps, kernel=kernel, transform_input=True, Y_bounds=Y_bounds
        )

        if check_post_mean and (
            (i % check_post_mean_every_k == 0) or (i == total_training_round - 1)
        ):
            if current_strategy in ["uncorrelated"]:
                use_mean = True
            else:
                use_mean = False
            # evaluate posterior mean after each iteration
            single_post_mean_X = gen_post_mean(
                outcome_model, pref_model, problem, use_mean=use_mean
            )
            post_mean_X.append(single_post_mean_X)
            post_mean_idx.append(i)

        run_time = time.time() - start_time
        run_times.append(run_time)
        print(
            f"iteration {i}: acquisition takes {acq_run_time:.1f}s; "
            f"total runtime: {run_time:.1f}s"
        )
    if check_post_mean:
        post_mean_X = torch.cat(post_mean_X, dim=0)

    return (
        train_X,
        train_Y,
        train_comps,
        acq_run_times,
        run_times,
        post_mean_X,
        post_mean_idx,
        selected_pairs,
    )
