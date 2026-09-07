'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:00:38
Description: 
'''

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap
import os

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# detransformer = transforms.Compose([
#         transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
#         transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
#     ])


if __name__ == '__main__':
    opt = TestOptions().parse()

    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    model = create_model(opt)
    model.eval()


    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.3, det_size=(640,640),mode=mode)
    with torch.no_grad():

        pic_a_paths = opt.pic_a_path.split(",")

        identity_vectors = []

        for pic_a in pic_a_paths:

            print("Processing source:", pic_a)

            img_a_whole = cv2.imread(pic_a)

            if img_a_whole is None:
                raise ValueError(f"Could not read image: {pic_a}")
            
            print("Loaded:", img_a_whole is not None)

            if img_a_whole is not None:
                print("Shape:", img_a_whole.shape)


            result = app.get(img_a_whole,crop_size)

            if result is None:
                raise ValueError(f"No face detected/aligned in: {pic_a}")

            img_a_align_crop, _ = result

            img_a_align_crop_pil = Image.fromarray(
            cv2.cvtColor(
            img_a_align_crop[0],
            cv2.COLOR_BGR2RGB
            )
        )

            img_a = transformer_Arcface(
            img_a_align_crop_pil
            )

            img_id = img_a.unsqueeze(0)

            img_id = img_id.cpu()

            img_id_downsample = F.interpolate(
            img_id,
            size=(112, 112)
        )

            identity_vector = model.netArc(
            img_id_downsample
        )

    # Normalize each source embedding first
            identity_vector = F.normalize(
            identity_vector,
            p=2,
            dim=1
            )

            identity_vectors.append(identity_vector)


# Combine all source identities
        latend_id = torch.cat(
    identity_vectors,
    dim=0
).mean(
    dim=0,
    keepdim=True
)

# Normalize the averaged identity
        latend_id = F.normalize(
        latend_id,
    p=2,
    dim=1
    )
        
        # img_a = Image.open(pic_a).convert('RGB')
      
        

        # pic_b = opt.pic_b_path
        # img_b_whole = cv2.imread(pic_b)
        # img_b_align_crop, b_mat = app.get(img_b_whole,crop_size)
        # img_b_align_crop_pil = Image.fromarray(cv2.cvtColor(img_b_align_crop,cv2.COLOR_BGR2RGB)) 
        # img_b = transformer(img_b_align_crop_pil)
        # img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        # convert numpy to tensor
    
        # img_att = img_att.cuda()

        #create latent id
        
       

    print("Number of source images:", len(identity_vectors))
    print("Average Identity Shape:", latend_id.shape)
    print("Averaged Identity norm:", torch.norm(latend_id).item())

    video_swap(opt.video_path, latend_id, model, app, opt.output_path,temp_results_dir=opt.temp_path,\
            no_simswaplogo=opt.no_simswaplogo,use_mask=opt.use_mask,crop_size=crop_size)

