import torch
from torch import optim
from tqdm import tqdm
from torch.nn.functional import mse_loss
import math
from score_estimators import LEScore, IdealScore
import wandb
import torch.nn as nn
from torch.nn.parallel import parallel_apply
from torch import amp

def generate_patch_estimators(
          image_dim : int, 
          patch_sizes : list[int], 
          dataset, 
          schedule, 
          batch_size : int = 64,
          max_samples : int = 10000,
          conditional : bool = False
          ):
    """
    Generates a list of patch score estimators for each patch size
    """
    estimators = []
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    for i, patch_size in enumerate(patch_sizes):
        dev = devices[i % len(devices)]  
        if patch_size >= image_dim:
            estimator = IdealScore(dataset, schedule=schedule, max_samples=max_samples, conditional=conditional).to(dev)
            estimators.append(estimator)
            break # stop if the kernel size is larger than the image size

        estimator = LEScore(dataset, schedule=schedule, kernel_size=patch_size, 
                                  batch_size=batch_size, max_samples=max_samples, 
                                  conditional=conditional).to(dev)

        estimators.append(estimator)
    return estimators

def cosine_noise_schedule(t, mode='legacy'):
    if mode == 'legacy':
        return 1-torch.cos((t) / 1.008 * math.pi / 2) ** 2	
    # returns beta
    return 1-torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    
# Assumptions made: the dimension of the images are square hence will only be taking into account one of the height or width
def train_smallmodel(model, 
                    dataset,
                    train_loader,
                    noise_schedule, 
                    patch_sizes : int = [1,3,5],
                    batch_size : int = 64,
                    max_samples : int = 10000,
                    image_dim : int = 32,
                    max_t : int = 1000,
                    num_epochs : int = 100,
                    lr : float = 2e-4,
                    gamma : float = 0.99995,
                    wd : float = 0.001,
                    conditional : bool = False,
                    save_interval : int = 1,
                    checkpoint :str = './model_checkpoints/smallmodel',
                    args = False):
    """
    Train a small diffusion model by matching predicted weights to the ideal combination of score estimators over patch sizes.
    """
    model.train()
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    main_device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Initialise the patch size 
    estimators = generate_patch_estimators(image_dim, patch_sizes, dataset, noise_schedule, batch_size=batch_size, max_samples=max_samples, conditional=conditional)
    ideal_score_estimator = IdealScore(dataset, schedule=noise_schedule, max_samples=max_samples).to(main_device)

    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=True)
        
        for batch_num, (images, labels) in enumerate(loop):
            b, c, h, w = images.shape
            optimizer.zero_grad()
   
            images = images.to(main_device, non_blocking=True)
            labels = labels.to(main_device, non_blocking=True)

            t = torch.randint(0, max_t, (1,), device=main_device).float() / max_t 
       
            # Get noise level from schedule
            beta_t = noise_schedule(t) 
            
            noise = torch.normal(0,1,images.shape, device=main_device)
            noised_images = torch.sqrt(1 - beta_t)[:, None, None, None] * images + torch.sqrt(beta_t)[:, None, None, None] * noise # to go to this point in the forward pass

            """with torch.no_grad():
                # The ideal score for the noised image
                ideal_score_t = ideal_score_estimator(noised_images, t)
                patch_scores = []
                
                for i, estimator in enumerate(estimators):
                    dev = devices[i % len(devices)]  
                    x_dev = noised_images.to(dev)
                    t = t.to(dev)
                    scores_i = estimator(x_dev, t)
                    patch_scores.append(scores_i.to(main_device))

                scores = torch.stack(patch_scores, dim=1)"""

            with torch.no_grad(), amp.autocast('cuda'):
                # ideal score on cuda:0
                ideal_score_t = ideal_score_estimator(noised_images, t)

                # build (input, t) tuples for each estimator & device
                args = [
                    (
                        noised_images.to(devices[i], non_blocking=True),
                        t.to(devices[i],    non_blocking=True)
                    )
                    for i in range(len(estimators))
                ]

                # kick off _all_ the forwards at once
                outputs = parallel_apply(estimators, args)

                # bring them back and stack
                patch_scores = [out.to(main_device, non_blocking=True)
                                for out in outputs]
                scores = torch.stack(patch_scores, dim=1)

            if conditional:
                predicted_weights = model(noised_images, label=labels)
            else:
                predicted_weights = model(noised_images) # [B, S, C, H, W]
                
            predicted_score = torch.sum(predicted_weights * scores, dim=1) # [B, C, H, W]

            loss = mse_loss(predicted_score, ideal_score_t)
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

            if args.wandb:
                wandb.log({
                    "Loss": loss.item(),
                    "Epoch": epoch,
                    "Batch" : batch_num,
                    "Global Batch" : epoch * len(train_loader) + batch_num
                })
            
        scheduler.step()

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f"{checkpoint}_epoch{epoch}.pt")
            if wandb.run:
                wandb.save(f"{checkpoint}_epoch{epoch}.pt")