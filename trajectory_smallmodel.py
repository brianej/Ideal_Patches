import argparse
import os
import random
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import pandas as pd
from collections import OrderedDict

from small_model import SmallModel
from train_smallmodel import generate_patch_estimators, cosine_noise_schedule
from score_estimators import IdealScore
from utils.data import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model trajectory on small subset")
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--patch_sizes', nargs='+', type=int, default=[3, 7, 11])
    parser.add_argument('--steps', type=int, default=200, help='Number of steps')
    parser.add_argument('--delta', type=float, default=0.005, help='Delta')
    parser.add_argument('--wandb', action='store_true', help='Log to W&B')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_seed', type=int, default=123,help='Seed for fixed noise generation')
    parser.add_argument('--num_samples', type=int, default=32,help='Seed for fixed noise generation')
    parser.add_argument('--num_batches', type=int, default=10)
    return parser.parse_args()


def backward_step(score, points, beta_t, delta):
    noise = torch.randn_like(points)
    f_t = -1/2 * beta_t * points
    g_t = beta_t**0.5
    points = points - (f_t - beta_t * score) * delta + g_t * noise * delta**0.5
    return points


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_ids = [local_rank]

    dataset, metadata = get_dataset(args.dataset, train=False)

    input_shape = (metadata['num_channels'], metadata['image_size'], metadata['image_size'])
    model = SmallModel(input_shape, args.patch_sizes).to(device)
    state = torch.load('./model_checkpoints/smallmodel_epoch2_batch0-3-7.pt', weights_only=True)
    fixed_state = OrderedDict()
    for k, v in state.items():
        if k.startswith('module._orig_mod.'):
            new_key = k[len('module._orig_mod.'):]
        else:
            new_key = k
        fixed_state[new_key] = v

    model.load_state_dict(fixed_state)
    model.eval()

    estimators = generate_patch_estimators(metadata['image_size'], args.patch_sizes, dataset, cosine_noise_schedule, batch_size=args.num_batches, device=device)
    estimators = [torch.nn.DataParallel(est, device_ids=device_ids) for est in estimators]
    ideal_estimator = IdealScore(dataset, schedule=cosine_noise_schedule).to(device)
    ideal_estimator = torch.nn.DataParallel(ideal_estimator, device_ids=device_ids)

    mse = torch.nn.MSELoss(reduction='none')

    log_df = pd.DataFrame(columns=['Element', 'Timestep', 'Image', 'Loss', 'Weights', 'Predicted Score', 'Ideal Score'])

    # Generate a single noise tensor so that evaluations with different patch
    # sizes are comparable.  A dedicated seed ensures reproducibility
    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(args.noise_seed)

    samples_per_rank = args.num_samples // world_size

    if rank == 0:
        all_points = torch.rand(args.num_samples, metadata['num_channels'], metadata['image_size'], metadata['image_size'], generator=noise_gen, device=device)
        scatter_list = list(all_points.chunk(world_size))
    else:
        scatter_list = None

    points = torch.empty(samples_per_rank, metadata['num_channels'], metadata['image_size'], metadata['image_size'], device=device)
    dist.scatter(points, scatter_list=scatter_list, src=0)

    trajectories = torch.empty((args.steps+1, samples_per_rank, metadata['num_channels'], metadata['image_size'], metadata['image_size']))
    trajectories[0, :, :, :, :] = points

    if args.wandb:
        import wandb
        wandb.init(entity='brianej-personal', project='Ideal Patches',
                   name='Inference', config=vars(args))
        wandb.watch(model, log='all')

    # enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

    for step in range(1, args.steps+1):
        t_val = 1 - step * args.delta
        t = torch.tensor([t_val], device=device)
        beta_t = cosine_noise_schedule(t)

        ideal_score = ideal_estimator(points, t).to(device)

        scores = torch.zeros(samples_per_rank, len(args.patch_sizes), metadata['num_channels'], metadata['image_size'], metadata['image_size'], device=device)
        for i, est in enumerate(estimators):
            scores[:, i] = est(points, t).to(device)

        weights = model(points, t)
        predicted_score = torch.sum(weights * scores, dim=1)

        points = backward_step(predicted_score, points, beta_t, args.delta)
        trajectories[step, :, :, :, :] = points

        loss = mse(predicted_score, ideal_score)

        if args.wandb:
            images_to_log = [wandb.Image(img.detach().cpu()) for img in points]
            wandb.log({'step': step, 
                       'loss': loss.mean().item(),
                       'timestep': t_val, 
                       'images': images_to_log})

        for idx in range(samples_per_rank):
            log_df.loc[len(log_df)] = {
                'Element': idx,
                'Timestep': t_val,
                'Image': points[idx].detach().cpu().numpy(),
                'Loss': loss[idx].detach().cpu().numpy(),
                'Weights': weights[idx].detach().cpu().numpy(),
                'Predicted Score': predicted_score[idx].detach().cpu().numpy(),
                'Ideal Score': ideal_score[idx].detach().cpu().numpy()
            }

    # Finish W&B
    if args.wandb:
        wandb.finish()
        
    log_df.to_pickle(f"evaluation_logs3-7{rank}.pkl", compression="gzip")
    torch.save(trajectories, f"tensor-3-7-{rank}.pt")


if __name__ == '__main__':
    main()