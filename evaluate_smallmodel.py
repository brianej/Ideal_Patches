import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import random
import wandb
import pandas as pd

from small_model import SmallModel
from train_smallmodel import generate_patch_estimators, cosine_noise_schedule
from score_estimators import IdealScore
from utils.data import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SmallModel weights")
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_sizes', nargs='+', type=int, default=[3, 7, 11])
    parser.add_argument('--max_t', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--num_batches', type=int, default=10)
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

    # Load datasets
    test_dataset, metadata = get_dataset(args.dataset, train=False)
    sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=sampler)

    # Initialise model
    input_shape = (metadata['num_channels'], metadata['image_size'], metadata['image_size'])
    model = SmallModel(input_shape, args.patch_sizes).to(device)
    model = torch.compile(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    state_dict = torch.load("./model_checkpoints/smallmodel_epoch2_batch0-3-15-27.pt")
    model.load_state_dict(state_dict)
    model.eval()

    # Patch score estimators
    estimators = generate_patch_estimators(metadata['image_size'], args.patch_sizes,
                                          test_dataset, cosine_noise_schedule,
                                          batch_size=args.batch_size,
                                          device=device)
    
    ideal_estimator = IdealScore(test_dataset, schedule=cosine_noise_schedule).to(device)

    mse = torch.nn.MSELoss(reduction="none")

    # Initialise W&B
    if args.wandb:
        wandb.init(
            entity="brianej-personal",
            project="Ideal Patches", 
            group="Evaluate-3-15-27#2",
            config=vars(args)
        )

    if args.wandb:
        wandb.watch(model, log="all")

    step_logs = pd.DataFrame(
        columns=[
            "Batch",
            "Element",
            "Image",
            "Noised Image",
            "Loss",
            "Timestep",
            "Weights",
            "Predicted Score",
            "Ideal Score",
        ]
    )

    rank = dist.get_rank()

    for batch_idx, (images, _) in enumerate(test_loader):
        images = images.to(device)
        b, c, h, w = images.shape
        weights_nan = False
        ideal_score_nan = False
        scores_nan = False
        loss_nan = False

        # ---- safe per-image time-step sampling ----
        eps = 1e-4                                # keep away from 0 and 1
        t = torch.randint(0, args.max_t, (1,), device=device).float() / args.max_t 
        t = t * (1 - 2*eps) + eps                 # now t ∈ [eps, 1-eps]
        beta_t = cosine_noise_schedule(t)

        noise = torch.normal(0,1,images.shape, device=device)
        noised_images = torch.sqrt(1 - beta_t)[:, None, None, None] * images + torch.sqrt(beta_t)[:, None, None, None] * noise # to go to this point in the forward pass

        ideal_score = ideal_estimator(noised_images, t).to(device)

        scores = torch.zeros(b, len(args.patch_sizes), c, h, w, device=device)
        for i, estimator in enumerate(estimators):
            scores[:, i, :, :, :] = estimator(noised_images, t).to(device)

        weights = model(noised_images, t)
        predicted_score = torch.sum(weights * scores, dim=1)

        loss = mse(predicted_score, ideal_score)

        if torch.isnan(ideal_score).sum():
            print("Nan Detected in Ideal Score")
            ideal_score_nan = True
        if torch.isnan(scores).sum():
            print("Nan Detected in Scores")
            scores_nan = True
        if torch.isnan(weights).sum():
            print("Nan Detected in Weights")
            weights_nan = True
        if torch.isnan(loss).sum():
            print("Nan Detected in Loss")
            loss_nan = True

        if args.wandb:
            wandb.log({
                "Loss": loss.mean().item(),
                "t (Time)" : t[0].item(),
                "Weights Nan" : weights_nan,
                "Scores Nan" : scores_nan,
                "Ideal Score Nan" : ideal_score_nan,
                "Los Nan" : loss_nan
            })
        
        for img_idx in range(b):
            step_logs.loc[len(step_logs)] = {
                "Batch": batch_idx,
                "Element": img_idx,
                "Image" : images[img_idx].detach().cpu().numpy(),
                "Noised Image" : noised_images[img_idx].detach().cpu().numpy(),
                "Loss": loss[img_idx].detach().cpu().numpy(),
                "Timestep": t[0].item(),
                "Weights": weights[img_idx].detach().cpu().numpy(),
                "Predicted Score": predicted_score[img_idx].detach().cpu().numpy(),
                "Ideal Score": ideal_score[img_idx].detach().cpu().numpy(),
            }

    # Finish W&B
    if args.wandb:
        wandb.finish()
        
    step_logs.to_pickle(f"evaluation_logs3-7{rank}.pkl", compression="gzip")


if __name__ == '__main__':
    main()