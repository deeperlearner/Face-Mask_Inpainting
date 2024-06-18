import logging
import argparse
import logging
import os

import numpy as np
import torch
from torch import distributed
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb

from dataset_mask import MXFaceDataset
from Unet_I import UNet_I, Discriminator

# assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
# we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f'world_size={world_size} rank={rank}')
    distributed.init_process_group("nccl")
    print('distributed init_process_group done')
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gen_loss(gen,disc,binary,masked,adv_criterion,recon_criterion,lambda_recon):
    fake = gen(masked)
    pred = disc(fake)
    adv_loss = adv_criterion(pred,torch.ones_like(pred))
    # adv_loss += dice_loss(F.sigmoid(fake.squeeze(1)), binary.squeeze(1).float(), multiclass=False)
    recon_loss = recon_criterion(fake,binary)
    gen_loss = 0*adv_loss+(lambda_recon*recon_loss)
    return gen_loss


def get_disc_loss(disc,fake,binary,adv_criterion):
    fake_pred = disc(fake.detach())
    binary_pred = disc(binary)
    fake_loss = adv_criterion(fake_pred,torch.zeros_like(fake_pred))
    binary_loss = adv_criterion(binary_pred,torch.ones_like(binary_pred))
    disc_loss = (fake_loss+binary_loss)/2
    return disc_loss


def train_stage_I(save_model=False):
    batch_size = args.batch_size
    n_train = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    global_step = 0

    experiment = wandb.init(project='UNet stage I', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, save_checkpoint=save_model)
    )

    logging.info(f'''Starting training:
        Epochs:          {n_epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Checkpoints:     {save_model}
        Device:          {device.type}
    ''')
    
    stop_step = 10000
    for epoch in range(n_epochs):
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{n_epochs}', unit='img') as pbar:
            for masked, binary, label in dataloader:
                cur_batch_size = len(masked)
                masked = masked.to(device)
                binary = binary.to(device)
                binary = binary.permute(0, 3, 1, 2).float()
                
                # disc_opt.zero_grad()      ##update discriminator
                # with torch.no_grad():
                #     fake = gen_I(masked)
                # disc_loss = get_disc_loss(disc, fake, binary, adv_criterion)
                # disc_loss.backward(retain_graph=True)
                # disc_opt.step()
                
                gen_opt_I.zero_grad()
                gen_loss = get_gen_loss(gen_I, disc, binary, masked, adv_criterion, recon_criterion, lambda_recon)
                gen_loss.backward()
                gen_opt_I.step()

                pbar.update(masked.shape[0])
                global_step += 1
                experiment.log({
                    'step': global_step,
                    'Generator (U-Net) loss': gen_loss.item(),
                    # 'Discriminator loss': disc_loss.item(),
                    'epoch': epoch
                })
                # pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = 2000
                if global_step % division_step == 0:
                    with torch.no_grad():
                        fake = gen_I(masked)
                    fake_pred = (fake + 1) / 2
                    experiment.log({
                        'images': [
                            wandb.Image(masked[0].cpu()),
                        ],
                        'masks': {
                            'true': [
                                wandb.Image(binary[0].float().cpu()),
                            ],
                            'pred': [
                                wandb.Image(fake[0].float().cpu()),
                            ]
                        },
                        'step': global_step,
                        'epoch': epoch,
                    })
                    if save_model:
                        torch.save({'gen_I':gen_I.state_dict(),
                                    'gen_opt_I':gen_opt_I.state_dict(),
                                    # 'disc':disc.state_dict(),
                                    # 'disc_opt':disc_opt.state_dict(),
                                    }, f"models/UNet_I_{global_step}.pth")
                if global_step % stop_step == 0:
                    break


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--rec", type=str, help="rec file directory")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--dali", type=bool, default=False, help="use dali")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    args = parser.parse_args()

    seed = 2333
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f'torch version ={torch.__version__}')
    print(f'args.local_rank={args.local_rank}')
    print(f'world_size={world_size} rank={rank} local_rank={args.local_rank}')

    root_dir = args.rec
    local_rank = 0
    dataset = MXFaceDataset(root_dir, local_rank)

    # stage I
    adv_criterion = nn.BCEWithLogitsLoss()
    recon_criterion = nn.BCEWithLogitsLoss()#nn.L1Loss()
    lambda_recon = 200

    n_epochs = 1
    input_dim = 3
    binary_dim = 1
    learning_rate = 0.0002
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gen_I = UNet_I(input_dim, binary_dim).to(device)
    gen_opt_I = torch.optim.Adam(gen_I.parameters(), lr=learning_rate)
    disc = Discriminator(binary_dim).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=learning_rate)

    # print(count_parameters(gen_I))
    # os._exit(0)

    train_stage_I(save_model=True)
