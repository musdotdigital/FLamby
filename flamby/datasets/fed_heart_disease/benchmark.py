import argparse
from typing import List

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader as dl
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedHeartDisease,
    metric,
    NUM_CLIENTS
)
from flamby.utils import evaluate_model_on_tests


def model_available(path: str):
    try:
        torch.load(path)
        return True
    except Exception:
        return False


def main(num_workers_torch, log=False, log_period=10, debug=False, cpu_only=False, center=[0], pooled=True):
    """Function to execute the benchmark on Heart Disezase.

    Parameters
    ----------
    debug : bool
        Whether or not to use the dataset obtained in debug mode. Default to False.
        Used for consistency with other datasets' APIs.
    """

    print(
        f"Using torch.multiprocessing_workers: {num_workers_torch}, log: {log}, log_period: {log_period}, debug: {debug}, cpu_only: {cpu_only}, center: {center}")

    metrics_dict = {"AUC": metric}

    use_gpu = torch.has_mps and not (cpu_only)

    training_dl = dl(
        FedHeartDisease(train=True, pooled=pooled, debug=debug,
                        center=center),
        num_workers=num_workers_torch,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    def test_dl(center: List[int]):
        return dl(
            FedHeartDisease(train=False, pooled=pooled, debug=debug,
                            center=center),
            num_workers=num_workers_torch,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

    TRAINING_CENTER_STRING = ''.join(str(e) for e in center)
    OUTPUT_STRING = "pooled" if pooled else f"of center(s): {TRAINING_CENTER_STRING}"
    MODEL_PATH = 'models/' + f'model_{TRAINING_CENTER_STRING}.pt'

    print(
        f"The training set {OUTPUT_STRING} contains {len(training_dl.dataset)} records")

    if log:
        # We compute the number of batches per epoch
        num_local_steps_per_epoch = len(training_dl.dataset) // BATCH_SIZE
        num_local_steps_per_epoch += int(
            (len(training_dl.dataset) - num_local_steps_per_epoch * BATCH_SIZE) > 0
        )

    results = []
    seeds = np.arange(42, 43).tolist()

    for seed in seeds:
        # At each new seed we re-initialize the model
        # and training_dl is shuffled as well
        torch.manual_seed(seed)

        m = Baseline()

        if not model_available(MODEL_PATH):
            print('no model available, training...')
            # We put the model on GPU whenever it is possible
            device = "cpu"
            if use_gpu:
                device = torch.device("mps")
                m = m.to(device)

            loss = BaselineLoss()
            optimizer = optim.Adam(m.parameters(), lr=LR)

            if log:
                # We create one summarywriter for each seed in order to overlay the plots
                print('Creating tensorboard writer...', seed)
                writer = SummaryWriter(log_dir=f"./runs/seed{seed}")

            for e in tqdm(range(NUM_EPOCHS_POOLED)):
                if log:
                    # At each epoch we look at the histograms of all the network's parameters
                    for name, p in m.named_parameters():
                        writer.add_histogram(f"client_0/{name}", p, e)

                for s, (X, y) in enumerate(training_dl):
                    # traditional training loop with optional GPU transfer
                    if use_gpu:
                        X = X.to(device)
                        y = y.to(device)

                    optimizer.zero_grad()
                    y_pred = m(X)
                    lm = loss(y_pred, y)
                    lm.backward()
                    optimizer.step()

                    if log:
                        current_step = s + num_local_steps_per_epoch * e
                        if (current_step % log_period) == 0:
                            writer.add_scalar(
                                "Loss/train/client",
                                lm.item(),
                                s + num_local_steps_per_epoch * e,
                            )
                            for k, v in metrics_dict.items():
                                train_batch_metric = v(
                                    y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
                                )
                                writer.add_scalar(
                                    f"{k}/train/client",
                                    train_batch_metric,
                                    s + num_local_steps_per_epoch * e,
                                )

            # Print optimizer's state_dict
            print("Optimizer's state_dict:")
            for var_name in optimizer.state_dict():
                print(var_name, "\t", optimizer.state_dict()[var_name])

            torch.save(m, MODEL_PATH)

        else:
            m = torch.load(MODEL_PATH)

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in m.state_dict():
            print(param_tensor, "\t", m.state_dict()[param_tensor].size())

        current_results_dict = evaluate_model_on_tests(
            m, [test_dl([i]) for i in range(NUM_CLIENTS)], metric, use_gpu=use_gpu
        )

        print('current_results_dict', current_results_dict)

        for i in range(NUM_CLIENTS):
            results.append(current_results_dict[f"client_test_{i}"])

    results = np.array(results)
    print('results', results)

    if log:
        for i in range(results.shape[0]):
            writer = SummaryWriter(log_dir=f"./runs/tests_seed{seeds[i]}")
            writer.add_scalar("AUC-test", results[i], 0)

    print("Benchmark Results on Heart Disease pooled:")
    print(f"mAUC on {len(seeds)} runs: {results.mean(): .2%} \\pm {results.std(): .2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers-torch",
        type=int,
        help="How many workers to use for the batching.",
        default=8,
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Whether to activate tensorboard logging or not default to no logging",
    )
    parser.add_argument(
        "--log-period",
        type=int,
        help="The period in batches for the logging of metric and loss",
        default=10,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to use the dataset obtained in debug mode.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Deactivate the GPU to perform all computations on CPU only.",
    )

    parser.add_argument(
        "--centers",
        nargs='+',
        type=int,
        help="Which center(s) to generate model from.",
        default=[0],
    )

    parser.add_argument(
        "--pool",
        action="store_true",
        help="Whether to use the dataset obtained in debug mode.",
    )

    args = parser.parse_args()
    main(args.num_workers_torch, args.log, args.log_period,
         args.debug, args.cpu_only, args.centers, args.pool)
