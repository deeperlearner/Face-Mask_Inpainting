import logging
import argparse
import logging
import os
from math import exp

import numpy as np
import torch
from torch import distributed
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
from torchvision.models import inception_v3
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb
import cv2

from dataset_mask_v2 import MXFaceDataset
from Unet_I import UNet_I, Discriminator
from Unet_II import UNet_II, Discriminator_whole, Discriminator_mask, PerceptualNet

# assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
# we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def normalize(img):
    return (img-(-1))/(1-(-1))
def anti_normalize(img):
    return img*(1-(-1))+(-1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        ###
        img1 = (img1+1)/2
        img2 = (img2+1)/2
        ###
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def recon_loss(gt,fake,recon_criterion):
    ssim = SSIM()
    ssim_loss = ssim(gt,fake)
    l1_loss = recon_criterion(gt,fake)
    return l1_loss,ssim_loss


def percep_loss(gt,fake):
    percep_net = PerceptualNet()
    return percep_net(gt,fake)


def discwhole_loss_func(disc_whole,gt,mask,binary,fake,adv_criterion,lambda_Dwhole):
    input_imgs = torch.cat((mask,binary),1)
    fake_pred = disc_whole(fake.detach(),input_imgs)
    gt_pred = disc_whole(gt,input_imgs)
    fake_loss = adv_criterion(fake_pred,torch.zeros_like(fake_pred))
    gt_loss = adv_criterion(gt_pred,torch.ones_like(gt_pred))
    return lambda_Dwhole * (fake_loss+gt_loss)/2


def discmask_loss_func(disc_mask, gt,fake,mask,binary, adv_criterion, lambda_Dmask): 
    nor_mask = normalize(mask)
    nor_binary = normalize(binary)
    nor_fake = normalize(fake)
    
    oofs = torch.mul(nor_mask,1-nor_binary)
    oops = torch.mul(nor_fake,nor_binary)
    ooo = anti_normalize(oofs+oops)
    input_imgs = torch.cat((mask,binary),1)
    fake_pred = disc_mask(ooo.detach(),input_imgs)
    gt_pred = disc_mask(gt,input_imgs)
    
    fake_loss = adv_criterion(fake_pred,torch.zeros_like(fake_pred))
    gt_loss = adv_criterion(gt_pred,torch.ones_like(gt_pred))
    
    return lambda_Dmask * (fake_loss+gt_loss)/2


def gen_adv_loss(gen,disc, gt,mask,binary, adv_criterion):
    input_imgs = torch.cat((mask,binary),1)
    fake = gen(input_imgs)
    fake_pred = disc(fake,input_imgs)
    adv_loss = adv_criterion(fake_pred,torch.ones_like(fake_pred))
    return adv_loss,fake


def generator_loss(cur_step,gen,disc_whole,disc_mask, gt,mask,binary,
                  adv_criterion,recon_criterion,
                  lambda_recon,lambda_adv_whole,lambda_adv_mask):
    if cur_step<3516*6:
        adver_loss_whole,fake = gen_adv_loss(gen,disc_whole,gt,mask,binary,adv_criterion)
        l1_loss,ssim_loss = recon_loss(gt,fake,recon_criterion)
        reconstruction_loss = l1_loss*0.5 + (1-ssim_loss)*0.5
        perceptual_loss = percep_loss(gt,fake)
        gen_loss = lambda_recon*(reconstruction_loss+perceptual_loss)+lambda_adv_whole*adver_loss_whole
    else:
        adver_loss_whole,fake = gen_adv_loss(gen,disc_whole,gt,mask,binary,adv_criterion)
        adver_loss_mask,fake = gen_adv_loss(gen,disc_mask,gt,mask,binary,adv_criterion)
        l1_loss,ssim_loss = recon_loss(gt,fake,recon_criterion)
        reconstruction_loss = l1_loss*0.5 + (1-ssim_loss)*0.5
        perceptual_loss = percep_loss(gt,fake)
        gen_loss = lambda_recon*(reconstruction_loss+perceptual_loss)+lambda_adv_whole*adver_loss_whole+lambda_adv_mask*adver_loss_mask
    return gen_loss,fake,l1_loss,ssim_loss,perceptual_loss


def noise_removal(binary_mask):
    # Convert tensor to NumPy array and squeeze to remove the channel dimension
    image_np = binary_mask.squeeze(1).detach().cpu().numpy()  # Resulting shape: (batch, 256, 256)
    
    # Convert the NumPy array to range [0, 255]
    image_np = (image_np * 255).astype(np.uint8)
    
    # Define the structuring element (kernel)
    kernel = np.ones((5, 5), np.uint8)
    
    # Initialize an array to store the processed images
    processed_images = np.empty_like(image_np)
    
    # Process the images using OpenCV in a vectorized manner
    batch_size = binary_mask.size(0)
    for i in range(batch_size):
        # Apply thresholding
        ret, imgg = cv2.threshold(image_np[i], 220, 255, cv2.THRESH_BINARY)
        
        # Apply morphological opening
        opening = cv2.morphologyEx(imgg, cv2.MORPH_OPEN, kernel)
        
        # Store the processed image
        processed_images[i] = opening
    
    # Convert the processed NumPy array back to a PyTorch tensor
    result_tensor = torch.from_numpy(processed_images).unsqueeze(1).cuda().float() / 255.0  # Shape: (batch, 1, 256, 256)
    
    # print("Original Tensor shape:", binary_mask.shape)
    # print("Result Tensor shape:", result_tensor.shape)

    # # Make a grid of the original and processed images
    # original_grid = make_grid(binary_mask, nrow=4, normalize=False, scale_each=True)
    # processed_grid = make_grid(result_tensor, nrow=4, normalize=False, scale_each=True)
    
    # # Save the grid images
    # save_image(original_grid, 'original_images.png')
    # save_image(processed_grid, 'processed_images.png')

    return result_tensor


inception_model = inception_v3(pretrained=True)
inception_model.to(device)
inception_model = inception_model.eval() # Evaluation mode
inception_model.fc = torch.nn.Identity()

def matrix_sqrt(x):
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real,device=x.device)

def frechet_distance(mu_x,mu_y,sigma_x,sigma_y):
    return torch.norm(mu_x-mu_y)**2 + torch.trace(sigma_x+sigma_y-2*matrix_sqrt(sigma_x@sigma_y))

def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(),rowvar=False))


def train_stage_II(save_model=False):
    batch_size = args.batch_size
    mean_generator_loss = 0
    mean_disc_whole_loss = 0
    mean_disc_mask_loss = 0
    fake_features_list = []
    real_features_list = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    cur_step = 0  ##change##

    experiment = wandb.init(project='UNet stage II', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=n_epochs, batch_size=batch_size, learning_rate=lr, save_checkpoint=save_model)
    )

    for epoch in range(1,n_epochs+1):
        for face, masked_face, binary_mask_gt in tqdm(dataloader):
            face = face.to(device)
            masked_face = masked_face.to(device)

            with torch.no_grad():
                binary_mask = gen_I(masked_face)
                binary_mask = noise_removal(binary_mask)
            
                input_imgs = torch.cat((masked_face, binary_mask), 1)
                fake = gen_II(input_imgs)
                
            real_features = inception_model(face.to(device)).detach().to('cpu')    #FID
            real_features_list.append(real_features)
            fake_features = inception_model(fake.to(device)).detach().to('cpu')
            fake_features_list.append(fake_features)
            
            if cur_step%5==0:
                disc_whole_opt.zero_grad()
                disc_whole_loss = discwhole_loss_func(disc_whole, face, masked_face, binary_mask, fake, adv_criterion, lambda_Dwhole)
                disc_whole_loss.backward(retain_graph=True)
                disc_whole_opt.step()
                mean_disc_whole_loss += disc_whole_loss.item()/4
                
                if cur_step>=3516*6:
                    disc_mask_opt.zero_grad()
                    disc_mask_loss = discmask_loss_func(disc_mask, face, fake, masked_face, binary_mask, adv_criterion, lambda_Dmask)
                    disc_mask_loss.backward(retain_graph=True)
                    disc_mask_opt.step()
                    mean_disc_mask_loss += disc_mask_loss.item()/4
            
            
            gen_opt_II.zero_grad()
            gen_loss, fake, l1_loss, ssim_loss, perceptual_loss = generator_loss(
                cur_step, gen_II,disc_whole,disc_mask, 
                face, masked_face, binary_mask,
                adv_criterion, recon_criterion,
                lambda_recon, lambda_adv_whole, lambda_adv_mask
            )
            gen_loss.backward()
            gen_opt_II.step()
            mean_generator_loss += gen_loss.item()/20
            
            
            if cur_step%20 == 0:
                fake_features_all = torch.cat(fake_features_list)            #FID
                real_features_all = torch.cat(real_features_list)
                mu_fake = fake_features_all.mean(dim=0)
                mu_real = real_features_all.mean(dim=0)
                sigma_fake = get_covariance(fake_features_all)
                sigma_real = get_covariance(real_features_all)
                FID = frechet_distance(mu_real,mu_fake,sigma_real,sigma_fake).item()
                
                fid_file = open('FID_epoch16','w')       ##change##
                fid_file.write(str(cur_step)+"\n")
                fid_file.write(str(round(FID,4))+"\n"+"\n")
                fid_file.close()
                fake_features_list.clear()
                real_features_list.clear()

                loss_file = open('loss_epoch16','w')     ##change##
                loss_file.write(str(cur_step)+"\n")
                loss_file.write(str(round(mean_generator_loss,4))+"    "+str(round(mean_disc_whole_loss,4))+"    "+str(round(mean_disc_mask_loss,4))+"    ")
                loss_file.write(str(round(l1_loss.item(),4))+"    "+str(round(1-ssim_loss.item(),4))+"    "+str(round(perceptual_loss.item(),4)))
                loss_file.write("\n"+"\n")
                loss_file.close()
                mean_generator_loss = 0
                mean_disc_whole_loss = 0
                mean_disc_mask_loss = 0
                
            log_dict = {
                'step': cur_step,
                'Generator loss': gen_loss.item(),
            }
            if cur_step%5==0:
                log_dict.update({
                    'Discriminator (whole) loss': disc_whole_loss.item(),
                })
                if cur_step>=3516*6:
                    log_dict.update({
                        'Discriminator (mask) loss': disc_mask_loss.item(),
                    })
            experiment.log(log_dict)

            cur_step += 1
            display_step = 1000
            if cur_step % display_step == 0:
                fake_pred = (fake + 1) / 2
                experiment.log({
                    'face': [
                        wandb.Image(face.cpu()),
                    ],
                    'masked_face': [
                        wandb.Image(masked_face.cpu()),
                    ],
                    'fake(inpainted)': [
                        wandb.Image(fake_pred.cpu()),
                    ],
                    'step': cur_step,
                    'epoch': epoch,
                })
                if save_model:
                    torch.save({'gen_II':gen_II.state_dict(),
                               'gen_opt_II':gen_opt_II.state_dict(),
                               'disc_whole':disc_whole.state_dict(),
                               'disc_whole_opt':disc_whole_opt.state_dict(),
                               'disc_mask':disc_mask.state_dict(),
                               'disc_mask_opt':disc_mask_opt.state_dict()},
                              f"models/Inpaint_UNet_{cur_step}.pth")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--rec", type=str, help="rec file directory")
    parser.add_argument("--batch_size", type=int, default=3, help="batch size")
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
    input_dim = 3
    binary_dim = 1

    gen_I = UNet_I(input_dim, binary_dim).to(device)
    loaded_state = torch.load("models/UNet_I_10000.pth")
    gen_I.load_state_dict(loaded_state["gen_I"])
    gen_I.eval()

    # stage II
    adv_criterion = nn.BCEWithLogitsLoss()
    #adv_criterion = nn.MSELoss()
    recon_criterion = nn.L1Loss()
    lambda_recon = 100
    lambda_Dwhole = 0.3
    lambda_Dmask = 0.7
    lambda_adv_whole = 0.3
    lambda_adv_mask = 0.7

    n_epochs = 1
    input_dim = 4
    output_dim = 3
    disc_dim = 7
    lr = 0.0003

    gen_II = UNet_II(input_dim, output_dim).to(device)
    gen_opt_II = torch.optim.Adam(gen_II.parameters(), lr=lr)

    disc_whole = Discriminator_whole(disc_dim).to(device)
    disc_whole_opt = torch.optim.Adam(disc_whole.parameters(), lr=0.0001)

    disc_mask = Discriminator_mask(disc_dim).to(device)
    disc_mask_opt = torch.optim.Adam(disc_mask.parameters(), lr=0.0001)

    # print(count_parameters(gen_I))
    # print(count_parameters(gen_II))
    # os._exit(0)
    train_stage_II(save_model=True)
