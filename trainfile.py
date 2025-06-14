import argparse
import random
import torch
import wandb
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

from small_model import SmallModel
from train_smallmodel import train_smallmodel, cosine_noise_schedule, generate_patch_estimators
from utils.data import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train small diffusion model with optimal patch estimators")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--gamma', type=float, default=0.99995, help='LR scheduler decay')
    parser.add_argument('--wd', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--patch_sizes', nargs='+', type=int, default=[3,5])
    parser.add_argument('--max_samples', type=int, default=60000)
    parser.add_argument('--max_t', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--conditional', type=bool, default=False)
    parser.add_argument('--checkpoint', type=str, default='./model_checkpoints/smallmodel')
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    return parser.parse_args()


def main():
    args = parse_args()
    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Initialize W&B
    if args.wandb:
        wandb.init(
            entity="brianej-personal",
            project="Ideal Patches", 
            config=vars(args)
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and subset
    train_dataset, metadata = get_dataset(args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize model
    input_shape = (metadata['num_channels'], metadata['image_size'], metadata['image_size'])
    model = SmallModel(input_shape, args.patch_sizes)
    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    if args.wandb:
        wandb.watch(model, log="all")

    # Train
    trained = train_smallmodel(
        model=model,
        dataset=train_dataset,
        train_loader=train_loader,
        noise_schedule=cosine_noise_schedule,
        patch_sizes=args.patch_sizes,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        image_dim=metadata['image_size'],
        max_t=args.max_t,
        num_epochs=args.num_epochs,
        lr=args.lr,
        gamma=args.gamma,
        wd=args.wd,
        conditional=args.conditional,
        save_interval=args.save_interval,
        checkpoint=args.checkpoint,
        args=args
    )

    # Finish W&B
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()