import torch
from torch import optim
from tqdm import tqdm
from torch.nn.functional import mse_loss
import math
from score_estimators import LEScore, IdealScore
import wandb
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
            estimator = IdealScore(dataset, schedule=schedule, max_samples=max_samples).to(dev)
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
                    patch_sizes : list = [1,3,5],
                    batch_size : int = 64,
                    max_samples : int = 10000,
                    image_dim : int = 32,
                    max_t : int = 1000,
                    num_epochs : int = 100,
                    lr : float = 2e-4,
                    gamma : float = 0.99995,
                    wd : float = 0.001,
                    conditional : bool = False,
                    save_interval : int = 100,
                    checkpoint :str = './model_checkpoints/smallmodel',
                    args = False,
                    dist = None,
                    device = None):
    """
    Train a small diffusion model by matching predicted weights to the ideal combination of score estimators over patch sizes.
    """
    model.train()
    scaler = amp.GradScaler()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Initialise the patch size 
    estimators = generate_patch_estimators(image_dim, patch_sizes, dataset, noise_schedule, batch_size=batch_size, max_samples=max_samples, conditional=conditional)
    ideal_score_estimator = IdealScore(dataset, schedule=noise_schedule, max_samples=max_samples).to(device)

    for epoch in range(num_epochs):
        if dist is not None:
            train_loader.sampler.set_epoch(epoch)
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=True)
        
        for batch_num, (images, labels) in enumerate(loop):
            b, c, h, w = images.shape
            optimizer.zero_grad()
   
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ---- safe per-image time-step sampling ----
            eps = 1e-4                                # keep away from 0 and 1
            t = torch.randint(0, max_t, (1,), device=device).float() / max_t 
            t = t * (1 - 2*eps) + eps                 # now t ∈ [eps, 1-eps]

            with amp.autocast(device_type="cuda"):
                # Get noise level from schedule
                beta_t = noise_schedule(t) 
                
                noise = torch.normal(0,1,images.shape, device=device)
                noised_images = torch.sqrt(1 - beta_t)[:, None, None, None] * images + torch.sqrt(beta_t)[:, None, None, None] * noise # to go to this point in the forward pass

                # The ideal score for the noised image
                ideal_score_t = ideal_score_estimator(noised_images, t).to(device)
                scores = torch.zeros(b, len(patch_sizes), c, h, w, device=device) # [B, S, C, H, W]
                for i, estimator in enumerate(estimators):
                    scores_i = estimator(noised_images, t).to(device=device)
                    scores[:, i, :, :, :] = scores_i

                predicted_weights = model(noised_images, t) # [B, S, C, H, W]
                    
                predicted_score = torch.sum(predicted_weights * scores, dim=1) # [B, C, H, W]

                loss = mse_loss(predicted_score, ideal_score_t)

                scaler.scale(loss).backward()

            if torch.isnan(loss):
                print(f"[NaN skipped] iter={batch_num, epoch}  loss={loss.item():.4e}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.step(optimizer)    
            scaler.update()  

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

            if args.wandb:
                wandb.log({
                    "Loss": loss.item(),
                    "Epoch": epoch,
                    "Batch" : batch_num,
                    "Global Batch" : epoch * len(train_loader) + batch_num,
                    "t (Time)" : t[0]
                })
            
            if batch_num % save_interval == 0:
                torch.save(model.state_dict(), f"{checkpoint}_epoch{epoch}_batch{batch_num}_re.pt")
                if wandb.run:
                    wandb.save(f"{checkpoint}_epoch{epoch}_batch{batch_num}_re.pt")
            
        scheduler.step()

        torch.save(model.state_dict(), f"{checkpoint}_epoch{epoch}.pt")
        if wandb.run:
            wandb.save(f"{checkpoint}_epoch{epoch}.pt")