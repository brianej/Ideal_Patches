import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import random
import wandb

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
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--checkpoint', type=str, default='./model_checkpoints/smallmodel.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_batches', type=int, default=10)
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Initialise W&B
    if args.wandb:
        wandb.init(
            entity="brianej-personal",
            project="Ideal Patches", 
            group="Evaluate-",
            config=vars(args)
        )

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
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Patch score estimators
    estimators = generate_patch_estimators(metadata['image_size'], args.patch_sizes,
                                          test_dataset, cosine_noise_schedule,
                                          batch_size=args.batch_size,
                                          max_samples=args.max_samples)
    
    ideal_estimator = IdealScore(test_dataset, schedule=cosine_noise_schedule,
                                 max_samples=args.max_samples).to(device)

    mse = torch.nn.MSELoss()

    if args.wandb:
        wandb.watch(model, log="all")

    for batch_idx, (images, _) in enumerate(test_loader):
        if batch_idx >= args.num_batches:
            break
        images = images.to(device)
        b, c, h, w = images.shape

        eps = 1e-4
        t = torch.randint(0, args.max_t, (1,), device=device).float() / args.max_t
        t = t * (1 - 2 * eps) + eps
        beta_t = cosine_noise_schedule(t)
        noise = torch.randn_like(images)
        noised = torch.sqrt(1 - beta_t)[:, None, None, None] * images + torch.sqrt(beta_t)[:, None, None, None] * noise

        ideal_score = ideal_estimator(noised, t)

        scores = torch.zeros(b, len(args.patch_sizes), c, h, w, device=device)
        for i, estimator in enumerate(estimators):
            scores[:, i] = estimator(noised, t).to(device)

        weights = model(noised, t)
        predicted_score = torch.sum(weights * scores, dim=1)

        loss = mse(predicted_score, ideal_score)

        if args.wandb:
            wandb.log({
                "Loss": loss.item(),
                "t (Time)": t[0],
                "weights": weights.detach().cpu(),
                "Predicted Score": predicted_score.detach().cpu(),
                "Ideal Score": ideal_score.detach().cpu()
            })

    # Finish W&B
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
