import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from skimage.io import imsave
from train_scripts.dib_utils import  MeanShift, Vgg16, gram_matrix
import os
import cv2
from tqdm import tqdm
import shutil

def get_input_optimizer(first_pass_img):
    optimizer = optim.LBFGS([first_pass_img.requires_grad_()])
    return optimizer

def create_folder(name):
    if not os.path.exists(name):
        os.mkdir(name)

def step_2(images, output_dir, split):
    create_folder(f"{output_dir}/{split}")
    create_folder(f"{output_dir}/{split}/images")
    create_folder(f"{output_dir}/{split}/labels")


    for file in tqdm(images):
        label_name = file.split(".")[0]+".txt"
        bg = file.split(".")[0][-3:]
        
        first_pass_img = np.array(Image.open(f"{INPUT_DIR}/{file}").convert('RGB').resize((ss, ss)))
        target_img = np.array(Image.open(f"back/{bg}.jpg").convert('RGB').resize((ts, ts)))
        first_pass_img = torch.from_numpy(first_pass_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(0)
        target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(0)

        first_pass_img = first_pass_img.contiguous()
        target_img = target_img.contiguous()
        first_pass_img_c = first_pass_img.clone().detach()

        optimizer = get_input_optimizer(first_pass_img)
        run = [0]

        while run[0] <= num_steps:
        
            def closure():
                
                # Compute Loss Loss    
                target_features_style = vgg(mean_shift(target_img))
                target_gram_style = [gram_matrix(y) for y in target_features_style]
                blend_features_style = vgg(mean_shift(first_pass_img))
                blend_gram_style = [gram_matrix(y) for y in blend_features_style]
                style_loss = 0
                for layer in range(len(blend_gram_style)):
                    style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
                style_loss /= len(blend_gram_style)  
                style_loss *= style_weight        
                
                # Compute Content Loss
                content_features = vgg(mean_shift(first_pass_img_c))
                content_loss = content_weight * mse(blend_features_style.relu2_2, content_features.relu2_2)

                # Compute TV Reg Loss
                tv_loss = torch.sum(torch.abs(first_pass_img[:, :, :, :-1] - first_pass_img[:, :, :, 1:])) + \
                        torch.sum(torch.abs(first_pass_img[:, :, :-1, :] - first_pass_img[:, :, 1:, :]))
                tv_loss *= tv_weight
                
                # Compute Total Loss and Update Image
                loss = style_loss + content_loss + tv_loss
                optimizer.zero_grad()
                loss.backward()
                
                run[0] += 1
                return loss
            
            optimizer.step(closure)
        # clamp the pixels range into 0 ~ 255
        first_pass_img.data.clamp_(0, 255)

        # Make the Final Blended Image
        input_img_np = first_pass_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]

        # Save image from the second pass
        imsave(os.path.join(f"{output_dir}/{split}/images", file), input_img_np.astype(np.uint8))
        shutil.copy(f"boxes/{label_name}", f"{output_dir}/{split}/labels/{label_name}")
        break



# Define Loss Functions
mse = torch.nn.MSELoss()

# Import VGG network for computing style and content loss
mean_shift = MeanShift(0)
vgg = Vgg16().to(0)


style_weight = 1e6; content_weight = 1e3; tv_weight = 1e-6
ss = 400; ts = 400
num_steps = 500


OUTPUT_DIR = "datasets/inpaint_dib"
INPUT_DIR = "inpaint_dib"

create_folder(OUTPUT_DIR)
    
files = os.listdir(INPUT_DIR)
files = [i for i in files if "box" not in i]
train_split = int(len(files)*0.9)
create_folder(f"datasets/inpaint_dib")

step_2(files[:train_split], OUTPUT_DIR, "train")
step_2(files[train_split:], OUTPUT_DIR, "valid")








    
