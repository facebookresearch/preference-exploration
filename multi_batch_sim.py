import os
import pickle
import random
import time
import warnings
from copy import deepcopy

import fire
import torch
from gpytorch.utils.warnings import NumericalWarning

from pbo import gen_pbo_candidates, get_pbo_pe_comparisons
from sim_helpers import (fit_outcome_model, fit_pref_model,
                         gen_parego_candidates, gen_pref_candidates_eval,
                         gen_true_util_data, run_one_round_sim)
from test_functions import gen_rand_X, problem_setup

warnings.filterwarnings(
    "ignore",
    message="Could not update `train_inputs` with transformed inputs",
)


def run_multi_batch_sim(problem_str, noisy, init_seed, kernel, comp_noise_type, tkwargs):
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

    large_batch_size = False

    if comp_noise_type == "constant":
        comp_noise = 0.1
    elif comp_noise_type == "probit":
        comp_noise = probit_noise
    elif comp_noise_type == "none":
        comp_noise_type = "constant"
        comp_noise = 0.0
    else:
        raise RuntimeError("Invalid comp_noise_type! Must be constant, probit, or none")

    if X_dim <= 5:
        init_n_outcome = 16
        gen_batch_size = 8
        # keep the top batch_size in the generated batch to simulate cherry picking
        batch_size = 8
    else:
        init_n_outcome = 32
        gen_batch_size = 16
        batch_size = 16
    n_batch = 3
    keep_winner_prob = None

    if large_batch_size:
        init_n_outcome = init_n_outcome * 2
        gen_batch_size = gen_batch_size * 2
        batch_size = batch_size * 2

    print("START MULTI-BATCH SIM")
    print(f"init_n_outcome: {init_n_outcome}")
    print(f"gen_batch_size: {gen_batch_size}")
    print(f"batch_size: {batch_size}")
    print(f"n_batch: {n_batch}")
    print(f"comp_noise_type: {comp_noise_type}, comp_noise: {comp_noise}")

    # (training method, next batch generating methods, if one-shot, sample_outcome)
    policies = [
        # Camera ready baselines
        # PBO
        ("pbo_ei_pe_eubo", None, False, False),  # PBO EUBO
        ("pbo_ei_pe_ts", None, False, False),  # PBO TS
        # EUBO family
        ("ri_eubo_y", "qnei", False, False),  # EUBO-y0
        ("ri_eubo_one_sample", "qnei", False, False),  # EUBO-zeta
        ("ri_eubo_rff", "qnei", False, False),  # EUBO-f
        # BALD family
        ("ri_bald_rff", "qnei", False, False),  # BALD-f
        ("ri_bald_yspace", "qnei", False, False),  # BALD-Y0
        # Random baselines
        ("random_ps", "qnei", False, True),  # Random-f
        ("uncorrelated", "qnei", False, None),  # Random-Y0
        # Non-PE baselines
        ("parego_only", "qnei", False, False),  # MOBO
        ("random_exp", None, False, False),  # Random experiment
        ("true_util_seq", "qnei", False, None),  # True utility
        # ======= one-shot baseilnes =========
        # PBO
        ("pbo_ei_pe_ts", None, True, False),
        ("pbo_ei_pe_eubo", None, True, False),
        # EUBO family
        ("ri_eubo_y", "qnei", True, False),
        ("ri_eubo_one_sample", "qnei", True, False),
        ("ri_eubo_rff", "qnei", True, False),
        # BALD family
        ("ri_bald_rff", "qnei", True, False),
        ("ri_bald_yspace", "qnei", True, False),
        # Random baselines
        ("random_ps", "qnei", True, True),  # Random f tilde
        ("uncorrelated", "qnei", True, None),  # Random-Y
    ]

    # shuffle the order in case experiments auto-restarted
    # so that we have roughly even number of exp run for each config
    random.shuffle(policies)

    output_filepath = (
        f"data/sim_results/multi_batch/interactive/sim_{problem_str}_{kernel}.pickle"
        # f"data/sim_results/multi_batch/one_shot/sim_{problem_str}_{kernel}.pickle"
        # f"data/sim_results/multi_batch/probit_noise/sim_{problem_str}_{kernel}.pickle"
    )

    # ========= start simulation from here ==========
    # initial observation
    init_outcome_X = fixed_init_X_dict[problem_prefix][init_seed].to(Y_bounds)
    init_outcome_Y = problem(init_outcome_X)

    for policy, gen_method, one_shot, sample_outcome in policies:
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
                sr["policy"],
                sr["one_shot"],
                sr["comp_noise_type"],
                sr["comp_noise"],
            )
            exp_set.add(past_exp_signature)

        curr_exp_signature = (init_seed, problem_str, policy, one_shot, comp_noise_type, comp_noise)
        if curr_exp_signature not in exp_set:
            exp_set.add(curr_exp_signature)
        else:
            print(f"Experiment {curr_exp_signature} was previously run! Skipping...")
            continue

        # total training round per preference session (or total rounds for one-shot case)
        n_pe_session_comp = 25
        total_training_round = (n_pe_session_comp * 3) if one_shot else n_pe_session_comp

        keep_winner_prob = None
        all_run_times = []
        all_acq_run_times = []
        print(policy, gen_method, one_shot, sample_outcome)

        outcome_X = init_outcome_X.clone()
        outcome_Y = init_outcome_Y.clone()

        # X/Y/comps for preference model
        # not used for Parego only policy
        train_X = None
        train_Y = None
        train_comps = None
        init_strategy = None
        learn_strategy = None

        init_round = Y_dim * 2

        if policy in (
            "parego_only",
            "qnehvi",
            "true_util_seq",
            "pbo_ts",
            "pbo_ei",
            "pbo_ei_pe_ts",
            "pbo_ei_pe_eubo",
            "random_exp",
        ):
            keep_winner_prob = -1
            for batch_i in range(n_batch):
                print(
                    f"init_seed {init_seed} - {policy}, one-shot: {one_shot}, gen_method: {gen_method}, batch: {batch_i}, on {problem_str}, total_training_round: {total_training_round}",
                )
                outcome_model = fit_outcome_model(outcome_X, outcome_Y, X_bounds=problem.bounds)

                X_baseline = outcome_X
                # do not take Y here because the returned Y is a posterior sample
                acq_start_time = time.time()
                if policy == "parego_only":
                    outcome_cand_X, _, _, _, = gen_parego_candidates(
                        model=outcome_model,
                        X=X_baseline,
                        Y=outcome_Y,
                        q=gen_batch_size,
                        problem=problem,
                        get_util=get_util,
                        comp_noise_type=comp_noise_type,
                        comp_noise=comp_noise,
                        sample_outcome=sample_outcome,
                        gen_method=gen_method,
                    )
                elif policy == "true_util_seq":
                    outcome_cand_X, _, _, _, = gen_true_util_data(
                        model=outcome_model,
                        X=X_baseline,
                        Y=outcome_Y,
                        q=gen_batch_size,
                        problem=problem,
                        get_util=get_util,
                        comp_noise_type=comp_noise_type,
                        comp_noise=comp_noise,
                        gen_method=gen_method,
                    )
                elif policy[:3] == "pbo":
                    utils = get_util(outcome_Y)
                    if (not one_shot) or (one_shot and batch_i == 0):
                        if policy in ("pbo_ts", "pbo_ei"):
                            pe_strategy = "random"
                        elif policy == "pbo_ei_pe_ts":
                            pe_strategy = "ts"
                        elif policy == "pbo_ei_pe_eubo":
                            pe_strategy = "eubo"
                        else:
                            raise ValueError("Unsupported PE strategy for PBO")

                        train_comps = get_pbo_pe_comparisons(
                            outcome_X,
                            train_comps,
                            problem,
                            utils,
                            init_round,
                            total_training_round,
                            comp_noise_type,
                            comp_noise,
                            pe_strategy=pe_strategy,
                        )

                        observed_comp_error_rate = (
                            (utils[train_comps][..., 0] < utils[train_comps][..., 1])
                            .float()
                            .mean()
                            .item()
                        )
                        print(f"observed_comp_error_rate:{observed_comp_error_rate:.3f}")
                    else:
                        print("not doing comparisons for PBO one-shot after init batch")
                    print("train_comps shape:", train_comps.shape)

                    if policy == "pbo_ts":
                        pbo_gen_method = "ts"
                    elif policy[:6] == "pbo_ei":
                        pbo_gen_method = "ei"
                    else:
                        raise ValueError("Unknown PBO policy!")
                    outcome_cand_X = gen_pbo_candidates(
                        outcome_X=outcome_X,
                        train_comps=train_comps,
                        q=gen_batch_size,
                        problem=problem,
                        pbo_gen_method=pbo_gen_method,
                    )
                elif policy == "random_exp":
                    outcome_cand_X = gen_rand_X(gen_batch_size, problem)
                else:
                    raise ValueError("Unknown baseline policy")
                acq_runtime = time.time() - acq_start_time
                print(f"{policy} candidate gen time: {acq_runtime:.2f}s")

                # noisy observation
                outcome_cand_Y = problem(outcome_cand_X)
                outcome_cand_util = get_util(outcome_cand_Y)

                # select top candidates
                select_idx = outcome_cand_util.topk(k=batch_size).indices
                outcome_cand_X = outcome_cand_X[select_idx, :]
                outcome_cand_Y = outcome_cand_Y[select_idx, :]

                outcome_X = torch.cat((outcome_X, outcome_cand_X))
                outcome_Y = torch.cat((outcome_Y, outcome_cand_Y))
        else:
            print(
                f"init_seed {init_seed} - {policy}, {gen_method}, one-shot: {one_shot}, on {problem_str}, total_training_round: {total_training_round}",
            )
            train_X = None
            train_Y = None
            train_comps = None
            selected_pairs = []
            keep_winner_prob = 0
            if policy == "ri_bald_yspace":
                init_strategy = "uncorrelated"
                learn_strategy = "bald_yspace"
            elif policy == "ri_bald_correct":
                init_strategy = "random_ps"
                learn_strategy = "bald_correct"
            elif policy == "ri_bald_rff":
                init_strategy = "random_ps"
                learn_strategy = "bald_rff"
            elif policy == "ri_eubo_rff":
                init_strategy = "random_ps"
                learn_strategy = "eubo_rff"
            elif policy == "ri_eubo_y":
                init_strategy = "uncorrelated"
                learn_strategy = "eubo_y"
            elif policy == "ri_eubo_one_sample":
                init_strategy = "random_ps"
                learn_strategy = "eubo_one_sample"
            elif policy == "ri_bald":
                init_strategy = "random_ps"
                learn_strategy = "bald"
            elif policy == "random":
                init_strategy = "random_ps"
                learn_strategy = "random"
            elif policy == "random_ps":
                init_strategy = "random_ps"
                learn_strategy = "random_ps"
            elif policy == "uncorrelated":
                init_strategy = "uncorrelated"
                learn_strategy = "uncorrelated"
            else:
                raise RuntimeError("Unsupported learning policy")

            if one_shot:
                outcome_model = fit_outcome_model(outcome_X, outcome_Y, X_bounds=problem.bounds)
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
                    train_X=train_X,
                    train_Y=train_Y,
                    train_comps=train_comps,
                    init_strategy=init_strategy,
                    learn_strategy=learn_strategy,
                    gen_method=gen_method,
                    keep_winner_prob=keep_winner_prob,
                    sample_outcome=sample_outcome,
                    kernel=kernel,
                    check_post_mean=False,
                    check_post_mean_every_k=5,
                    tkwargs=tkwargs,
                    selected_pairs=selected_pairs,
                )
                pref_model = fit_pref_model(
                    train_Y,
                    train_comps,
                    kernel=kernel,
                    transform_input=True,
                    Y_bounds=Y_bounds,
                )
                all_acq_run_times = all_acq_run_times + acq_run_times
                all_run_times = all_run_times + run_times

            for batch_i in range(n_batch):
                outcome_model = fit_outcome_model(outcome_X, outcome_Y, X_bounds=problem.bounds)
                if not one_shot:
                    current_batch_init_round = max(0, init_round - batch_i * total_training_round)
                    print(f"batch {batch_i}, current_batch_init_round: {current_batch_init_round}")
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
                        init_round=current_batch_init_round,
                        problem_str=problem_str,
                        noisy=noisy,
                        comp_noise_type=comp_noise_type,
                        comp_noise=comp_noise,
                        outcome_model=outcome_model,
                        outcome_X=outcome_X,
                        outcome_Y=outcome_Y,
                        train_X=train_X,
                        train_Y=train_Y,
                        train_comps=train_comps,
                        init_strategy=init_strategy,
                        learn_strategy=learn_strategy,
                        gen_method=gen_method,
                        keep_winner_prob=keep_winner_prob,
                        sample_outcome=sample_outcome,
                        kernel=kernel,
                        check_post_mean=False,
                        check_post_mean_every_k=5,
                        tkwargs=tkwargs,
                        selected_pairs=selected_pairs,
                    )

                    pref_model = fit_pref_model(
                        train_Y,
                        train_comps,
                        kernel=kernel,
                        transform_input=True,
                        Y_bounds=Y_bounds,
                    )
                    all_acq_run_times = all_acq_run_times + acq_run_times
                    all_run_times = all_run_times + run_times

                utils = get_util(train_Y)
                observed_comp_error_rate = (
                    (utils[train_comps][..., 0] < utils[train_comps][..., 1])
                    .float()
                    .mean()
                    .item()
                )
                print(f"pref observed_comp_error_rate: {observed_comp_error_rate:.3f}")

                # generate next batch candidate
                # for baselines, only consider outcome_X as train_X are never observed for real
                # and could be a good point
                X_baseline = outcome_X
                acq_start_time = time.time()
                outcome_cand_X, _ = gen_pref_candidates_eval(
                    outcome_model=outcome_model,
                    pref_model=pref_model,
                    X_baseline=X_baseline,
                    problem=problem,
                    gen_method=gen_method,
                    q=gen_batch_size,
                    tkwargs=tkwargs,
                )
                acq_runtime = time.time() - acq_start_time
                print(f"pref candidate gen time: {acq_runtime:.2f}s")
                # noisy observation
                outcome_cand_Y = problem(outcome_cand_X)
                outcome_cand_util = get_util(outcome_cand_Y)

                # select top candidates
                select_idx = outcome_cand_util.topk(k=batch_size).indices
                outcome_cand_X = outcome_cand_X[select_idx, :]
                outcome_cand_Y = outcome_cand_Y[select_idx, :]

                outcome_X = torch.cat((outcome_X, outcome_cand_X))
                outcome_Y = torch.cat((outcome_Y, outcome_cand_Y))

        train_X = None if train_X is None else deepcopy(train_X).detach().cpu()
        train_Y = None if train_Y is None else deepcopy(train_Y).detach().cpu()
        train_comps = None if train_comps is None else deepcopy(train_comps).detach().cpu()

        single_result = {
            "init_seed": init_seed,
            "problem_str": problem_str,
            "policy": policy,
            "kernel": kernel,
            "noise_std": problem.noise_std,
            "init_round": init_round,
            "total_training_round": total_training_round,
            "one_shot": one_shot,
            "run_times": all_run_times,
            "acq_run_times": all_acq_run_times,
            # method for generating candidates, qnei or ts
            "gen_method": gen_method,
            "init_strategy": init_strategy,
            "learn_strategy": learn_strategy,
            "keep_winner_prob": keep_winner_prob,
            "comp_noise_type": comp_noise_type,
            "comp_noise": comp_noise,
            "sample_outcome": sample_outcome,
            "init_n_outcome": init_n_outcome,
            "gen_batch_size": gen_batch_size,
            "batch_size": batch_size,
            "n_batch": n_batch,
            "outcome_X": deepcopy(outcome_X).detach().cpu(),
            "outcome_Y": deepcopy(outcome_Y).detach().cpu(),
            "train_X": train_X,
            "train_Y": train_Y,
            "train_comps": train_comps,
            "device": str(tkwargs["device"]),
            "dtype": str(tkwargs["dtype"]),
        }

        if os.path.isfile(output_filepath):
            sim_results = pickle.load(open(output_filepath, "rb"))
        else:
            sim_results = []
        sim_results.append(single_result)
        pickle.dump(sim_results, open(output_filepath, "wb"))
        torch.cuda.empty_cache()


def main(problem_str, noisy, init_seed, kernel, comp_noise_type, device):
    """
    Args:
        problem_str: problem string. see definition in test_functions.py
        noisy: whether we have noisy observation of the resopnse surface
        init_seed: initialization seed
        kernel: "default" (RBF) or "linear" (might not be numerically stable)
        comp_noise_type: "constant" or "probit"
    """
    assert isinstance(noisy, bool)

    dtype = torch.double
    if device == "cpu":
        device = torch.device("cpu")
    else:
        # Does not really work. Need to set the env var in command line.
        # os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
        # "device": torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"),
        # set device env variable externally
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tkwargs = {
        "dtype": dtype,
        "device": device,
    }

    warnings.filterwarnings("ignore", category=NumericalWarning)
    run_multi_batch_sim(
        problem_str=problem_str,
        noisy=noisy,
        init_seed=init_seed,
        kernel=kernel,
        comp_noise_type=comp_noise_type,
        tkwargs=tkwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)
