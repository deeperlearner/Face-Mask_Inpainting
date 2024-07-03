import sys
import os

import torch
from torchvision.utils import make_grid, save_image
import numpy as np
import cv2

from Face_Mask_Inpainting.Unet_I import UNet_I, Discriminator
from Face_Mask_Inpainting.Unet_II import UNet_II, Discriminator_whole, Discriminator_mask, PerceptualNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# stage I
input_dim = 3
binary_dim = 1

gen_I = UNet_I(input_dim, binary_dim).to(device)
loaded_state = torch.load("Face_Mask_Inpainting/models/UNet_I_10000.pth")
gen_I.load_state_dict(loaded_state["gen_I"])
gen_I.eval()

# stage II
input_dim = 4
output_dim = 3
gen_II = UNet_II(input_dim, output_dim).to(device)
loaded_state = torch.load("Face_Mask_Inpainting/models/Inpaint_UNet_2000.pth")
gen_II.load_state_dict(loaded_state["gen_II"])


# torch.set_printoptions(profile="full")
def inpaint(masked_face, local_labels, mode='training'):
    # print(masked_face.size())  # (256, 3, 112, 112)
    with torch.no_grad():
        I_mask = gen_I(masked_face)
        I_mask = noise_removal(I_mask)
        count_ones = torch.sum(I_mask, dim=(1, 2, 3))
        total_elements = I_mask.size(1) * I_mask.size(2) * I_mask.size(3)
        wear_mask = (count_ones / total_elements) >= 0.1  # mask coverage larger than 10%

        # Masked-face
        selected_I_mask = I_mask[wear_mask]
        selected_masked_face = masked_face[wear_mask]
        if mode == 'training':
            selected_labels = local_labels[wear_mask]

        # Non-masked face
        unselected_I_mask = I_mask[~wear_mask]
        unselected_masked_face = masked_face[~wear_mask]
        if mode == 'training':
            unselected_labels = local_labels[~wear_mask]

        # using generator to do inpainting
        input_imgs = torch.cat((selected_masked_face, selected_I_mask), 1)
        inpainted_img = gen_II(input_imgs)

        # # compute right inpainted face according to formula
        # postprocess_inpainted = selected_masked_face * (1 - selected_I_mask) + inpainted_img * selected_I_mask

        # concatenate with face without mask
        batch_img = torch.cat((inpainted_img, unselected_masked_face), 0)
        batch_mask = torch.cat((selected_I_mask, unselected_I_mask), 0)
        if mode == 'training':
            batch_labels = torch.cat((selected_labels, unselected_labels), 0)

        # 4-channels image
        RGBM_img = torch.cat((batch_img, batch_mask), 1)

    # # Make a grid of the original and processed images
    # masked_face_grid = make_grid(masked_face, nrow=8, normalize=False, scale_each=True)
    # inpainted_grid = make_grid(batch_img, nrow=8, normalize=False, scale_each=True)
    # I_mask_grid = make_grid(batch_mask, nrow=8, normalize=False, scale_each=True)
    
    # # Save the grid images
    # save_image(masked_face_grid, 'masked_face.png')
    # save_image(inpainted_grid, 'inpainted.png')
    # save_image(I_mask_grid, 'I_mask.png')
    # os._exit(0)
    if mode == 'training':
        return RGBM_img, batch_labels
    elif mode == 'testing':
        return RGBM_img


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
