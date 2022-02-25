import os
import pickle
import random
import warnings
from copy import deepcopy

import fire
import torch
from gpytorch.utils.warnings import NumericalWarning

from sim_helpers import fit_outcome_model, run_one_round_sim
from test_functions import problem_setup

warnings.filterwarnings(
    "ignore",
    message="Could not update `train_inputs` with transformed inputs",
)


def run_pref_sim(problem_str, noisy, init_seed, gen_method, kernel, comp_noise_type, tkwargs):
    problem_prefix = "_".join(problem_str.split("_")[:2])
    fixed_init_X_dict = pickle.load(open("fixed_init_X_dict.pickle", "rb"))
    (
        X_dim,
        Y_dim,
        problem,
        util_type,
        get_util,
        Y_bounds,
        probit_noise,
    ) = problem_setup(problem_str, noisy=noisy, **tkwargs)

    # init_strategy, learn_strategy, keep_winner_prob, sample_outcome
    # if keep_winner_prob is None, we never enter learning phase
    exp_configs = [
        # EUBO family
        ("random_ps", "eubo_one_sample", 0, False),
        ("random_ps", "eubo_rff", 0, False),
        ("uncorrelated", "eubo_y", 0, False),
        # BALD family
        ("random_ps", "bald_rff", 0, False),
        ("uncorrelated", "bald_yspace", 0, False),
        # Random baselines
        ("random_ps", "random_ps", 0, True),
        ("uncorrelated", "uncorrelated", 0, None),
    ]

    # shuffle the order in case experiments auto-restarted
    # so that we have roughly even number of exp run for each config
    random.shuffle(exp_configs)

    output_filepath = (
        f"data/sim_results/within_session/camera_ready/sim_{problem_str}_{kernel}.pickle"
    )

    if comp_noise_type == "constant":
        comp_noise = 0.1
    elif comp_noise_type == "probit":
        comp_noise = probit_noise
    elif comp_noise_type == "none":
        comp_noise_type = "constant"
        comp_noise = 0.0
    else:
        raise RuntimeError("Invalid comp_noise_type! Must be constant, probit, or none")

    large_batch_size = False
    total_training_round = 80
    init_round = Y_dim * 2

    # ========= start simulation from here ==========
    if X_dim <= 5:
        outcome_n = 16
    else:
        outcome_n = 32
    if large_batch_size:
        outcome_n = outcome_n * 2

    print(f"Running init_seed: {init_seed}, outcome_n: {outcome_n}")

    # initial observation
    outcome_X = fixed_init_X_dict[problem_prefix][init_seed].to(Y_bounds)
    outcome_Y = problem(outcome_X)

    outcome_model = fit_outcome_model(outcome_X, outcome_Y, X_bounds=problem.bounds)
    for init_strategy, learn_strategy, keep_winner_prob, sample_outcome in exp_configs:
        # read past experiments to check repeated experiments
        # re-read the file every time as it can be updated by other processses
        if os.path.isfile(output_filepath):
            past_sim_results = pickle.load(open(output_filepath, "rb"))
        else:
            past_sim_results = []
        exp_set = set()
        for sr in past_sim_results:
            past_exp_signature = (
                sr["init_seed"],
                sr["problem_str"],
                sr["init_strategy"],
                sr["learn_strategy"],
                sr["comp_noise_type"],
                sr["comp_noise"],
            )
            exp_set.add(past_exp_signature)

        curr_exp_signature = (
            init_seed,
            problem_str,
            init_strategy,
            learn_strategy,
            comp_noise_type,
            comp_noise,
        )
        if curr_exp_signature not in exp_set:
            exp_set.add(curr_exp_signature)
        else:
            print(f"Experiment {curr_exp_signature} was previously run! Skipping...")
            continue

        print(
            f"Running {init_strategy} {learn_strategy} {keep_winner_prob} {sample_outcome} on {problem_str}, init_seed: {init_seed}"
        )
        selected_pairs = []
        (
            train_X,
            train_Y,
            train_comps,
            acq_run_times,
            run_times,
            post_mean_X,
            post_mean_idx,
            selected_pairs,
        ) = run_one_round_sim(
            total_training_round=total_training_round,
            init_round=init_round,
            problem_str=problem_str,
            noisy=noisy,
            comp_noise_type=comp_noise_type,
            comp_noise=comp_noise,
            outcome_model=outcome_model,
            outcome_X=outcome_X,
            outcome_Y=outcome_Y,
            train_X=None,
            train_Y=None,
            train_comps=None,
            init_strategy=init_strategy,
            learn_strategy=learn_strategy,
            gen_method=gen_method,
            keep_winner_prob=keep_winner_prob,
            sample_outcome=sample_outcome,
            kernel=kernel,
            check_post_mean=True,
            check_post_mean_every_k=5,
            tkwargs=tkwargs,
            selected_pairs=selected_pairs,
        )

        if len(post_mean_X) != 0:
            real_eval_util = get_util(problem.evaluate_true(post_mean_X))
            post_mean_X = deepcopy(post_mean_X).detach().cpu()
            real_eval_util = deepcopy(real_eval_util).detach().cpu()
        else:
            post_mean_X = None
            real_eval_util = None

        single_result = {
            "problem_str": problem_str,
            "init_seed": init_seed,
            "kernel": kernel,
            "gen_method": gen_method,
            "init_round": init_round,
            "noise_std": problem.noise_std,
            "comp_noise_type": comp_noise_type,
            "comp_noise": comp_noise,
            "outcome_n": outcome_n,
            "init_strategy": init_strategy,
            "learn_strategy": learn_strategy,
            "keep_winner_prob": keep_winner_prob,
            "sample_outcome": sample_outcome,
            "outcome_X": deepcopy(outcome_X).detach().cpu(),
            "outcome_Y": deepcopy(outcome_Y).detach().cpu(),
            "train_X": deepcopy(train_X).detach().cpu(),
            "train_Y": deepcopy(train_Y).detach().cpu(),
            "train_comps": deepcopy(train_comps).detach().cpu(),
            "post_mean_idx": post_mean_idx,
            "eval_X": post_mean_X,
            "eval_util": real_eval_util,
            "acq_run_times": acq_run_times,
            "run_times": run_times,
        }

        if os.path.isfile(output_filepath):
            sim_results = pickle.load(open(output_filepath, "rb"))
        else:
            sim_results = []
        sim_results.append(single_result)
        pickle.dump(sim_results, open(output_filepath, "wb"))
        torch.cuda.empty_cache()


def main(problem_str, noisy, init_seed, gen_method, kernel, comp_noise_type, device):
    """
    Args:
        problem_str: problem string. see definition in test_functions.py
        noisy: whether inject noise into the test function
        init_seed: initialization seed
        gen_methods: acquisition functions used, one of "ts", "qnei", or "qei"
        kernel: "default" (RBF) or "linear" (might not be numerically stable)
        comp_noise_type: "constant" or "probit"
    """
    assert isinstance(noisy, bool)

    dtype = torch.double
    if device == "cpu":
        device = torch.device("cpu")
    else:
        # set device env variable externally
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tkwargs = {
        "dtype": dtype,
        "device": device,
    }

    warnings.filterwarnings("ignore", category=NumericalWarning)

    print(problem_str, bool(noisy), init_seed, gen_method, kernel)
    run_pref_sim(
        problem_str=problem_str,
        noisy=noisy,
        init_seed=init_seed,
        gen_method=gen_method,
        kernel=kernel,
        comp_noise_type=comp_noise_type,
        tkwargs=tkwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)
