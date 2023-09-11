from __future__ import absolute_import, division, print_function
import datetime
import cv2
import math
import natsort
import os
import sys
import glob
import argparse
import time
import io

import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import pynng
from pynng import nng

import torch
from torchvision import transforms, datasets

import deps.monodepth2.networks as networks
from deps.monodepth2.layers import disp_to_depth
from deps.monodepth2.utils import download_model_if_doesnt_exist

from seathru import *
from matplotlib import pyplot as plt

from CloseDepth import closePoint
from MapFusion import Scene_depth
from MapFusion import flag
from MapFusion import Scene_depth_fusion
from MapOne import max_R
from MapTwo import R_minus_GB
from BL import getAtomsphericLight
from getRefinedTramsmission import Refinedtransmission
# import matlab.engine
import warnings
warnings.filterwarnings(action='ignore')

def preprocess_MIPandR_depth_map(depths, additive_depth, multiply_depth):
    depths = ((depths - np.min(depths)) / (
                np.max(depths) - np.min(depths))).astype(np.float32)
    depths = multiply_depth * (1.0 - depths) + 2+ additive_depth
    return depths

def preprocess_monodepth_depth_map(depths, additive_depth, multiply_depth):
    depths = ((depths - np.min(depths)) / (
                np.max(depths) - np.min(depths))).astype(np.float32)
    depths = (multiply_depth * (1.0 - depths)) + additive_depth
    return depths

# def colorCheck(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l_channel, a_channel, b_channel = cv2.split(img)
#     h, w, _ = img.shape
#     da = a_channel.sum() / (h * w) - 128
#     db = b_channel.sum() / (h * w) - 128
#     histA = [0] * 256
#     histB = [0] * 256
#     for i in range(h):
#         for j in range(w):
#             ta = a_channel[i][j]
#             tb = b_channel[i][j]
#             histA[ta] += 1
#             histB[tb] += 1
#     msqA = 0
#     msqB = 0
#     for y in range(256):
#         msqA += float(abs(y - 128 - da)) * histA[y] / (w * h)
#         msqB += float(abs(y - 128 - db)) * histB[y] / (w * h)
#     result = math.sqrt(da * da + db * db) / math.sqrt(msqA * msqA + msqB * msqB)
#     return result

def MIPandR(img,dep_max):
    blockSize = 9
    n = 5
    print('Estimate the MIP and R depth...')
    R_map = max_R(img, blockSize)
    plt.imshow(R_map)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    mip_map = R_minus_GB(img, blockSize, R_map)
    plt.imshow(mip_map)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    print('Depth fusion...')
    ibla = Scene_depth(R_map, mip_map, img)
    ibla = Refinedtransmission(ibla, img)
    print('Calculate the nearest distance...')
    AtomLight = getAtomsphericLight(ibla, img)
    ibla = (ibla) * 255
    d_1 = closePoint(img, AtomLight)
    depths = preprocess_MIPandR_depth_map(ibla, d_1, dep_max)
    return depths

def run(image,out,dep,removD,args):

    device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # Load image and preprocess
    img = Image.fromarray(rawpy.imread(image).postprocess()) if args.raw else pil.open(image).convert('RGB')
    # img.thumbnail((args.size, args.size), Image.ANTIALIAS)
    original_width, original_height = img.size
    # img = exposure.equalize_adapthist(np.array(img), clip_limit=0.03)
    # img = Image.fromarray((np.round(img * 255.0)).astype(np.uint8))
    input_image = img.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    print('Preprocessed image', flush=True)

    # PREDICTION
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
    mapped_im_depths = ((disp_resized_np - np.min(disp_resized_np)) / (
            np.max(disp_resized_np) - np.min(disp_resized_np))).astype(np.float32)
    print("Processed image", flush=True)
    print('Loading image...', flush=True)
    print('Estimate the Mono2 depth...')
    DepthMono1 = preprocess_monodepth_depth_map(mapped_im_depths, 2.0, 10.0)
    depths1 = DepthMono1
    DepthMono2 = preprocess_monodepth_depth_map(mapped_im_depths, 2.0, 8.0)
    depths2 = DepthMono2
    #MIP+R深度图
    img11 = cv2.imread(image)
    temp1 = flag(img11)
    if temp1 != 1:
        DepthMipandR1 = MIPandR(img11,10)
        DepthMipandR1 = cv2.bilateralFilter(DepthMipandR1, d=0, sigmaColor=3, sigmaSpace=40)
        DepthMipandR2 = MIPandR(img11,8)
        DepthMipandR2 = cv2.bilateralFilter(DepthMipandR2, d=0, sigmaColor=3, sigmaSpace=40)
        depths1 = Scene_depth_fusion(DepthMipandR1,DepthMono1,temp1)
        depths2 = Scene_depth_fusion(DepthMipandR2,DepthMono2,temp1)

    # 基于不同的最大深度恢复图片
    recovered1 = run_pipeline(np.array(img) / 255.0, depths1,removD, args)
    sigma_est1 = estimate_sigma(recovered1, multichannel=True, average_sigmas=True) / 10.0
    recovered1 = denoise_tv_chambolle(recovered1, sigma_est1, multichannel=True)
    im1 = Image.fromarray((np.round(recovered1 * 255.0)).astype(np.uint8))

    recovered2 = run_pipeline(np.array(img) / 255.0, depths2,removD, args)
    sigma_est2 = estimate_sigma(recovered2, multichannel=True, average_sigmas=True) / 10.0
    recovered2 = denoise_tv_chambolle(recovered2, sigma_est2, multichannel=True)
    im2 = Image.fromarray((np.round(recovered2 * 255.0)).astype(np.uint8))


    niqe1 = getUCIQE(1,im1)
    niqe2 = getUCIQE(1,im2)
    if niqe1 < niqe2 :
        im1.save(out, format='png')
    else:
        im2.save(out, format='png')
    print('Done.')

    # eng = matlab.engine.start_matlab()
    # niqe1 = eng.zhibiao_niqe(out)
    # niqe2 = eng.zhibiao_niqe(out)
    # print('ni1:',niqe1)
    # print('ni2:', niqe2)
    # if niqe1 < niqe2 :
    #     plt.imsave(dep, depths1)
    #     im1.save(out, format='png')
    # else:
    #     plt.imsave(dep, depths2)
    #     im2.save(out, format='png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image', required=True, help='Input image')
    #parser.add_argument('--depth-map',help='Input depth map')
    parser.add_argument('--output', default='output.png', help='Output filename')
    parser.add_argument('--f', type=float, default=2.0, help='f value (controls brightness)')
    parser.add_argument('--l', type=float, default=0.5, help='l value (controls balance of attenuation constants)')
    parser.add_argument('--p', type=float, default=0.01, help='p value (controls locality of illuminant map)')
    parser.add_argument('--min-depth', type=float, default=0.0,
                        help='Minimum depth value to use in estimations (range 0-1)')
    parser.add_argument('--max-depth', type=float, default=1.0,
                        help='Replacement depth percentile value for invalid depths (range 0-1)')
    parser.add_argument('--spread-data-fraction', type=float, default=0.05,
                        help='Require data to be this fraction of depth range away from each other in attenuation estimations')
    parser.add_argument('--size', type=int, default=1280, help='Size to output')
    parser.add_argument('--monodepth-add-depth', type=float, default=2.0, help='单深度图的附加值')
    parser.add_argument('--monodepth-multiply-depth', type=float, default=10.0,
                        help='单深度图的乘法值')
    parser.add_argument('--model-name', type=str, default="mono_1024x320",
                        help='monodepth model name')
    # parser.add_argument('--output-graphs', action='store_true', help='Output graphs')
    parser.add_argument('--raw', action='store_true', help='RAW image')
    args = parser.parse_args()

    folder = "."
    path = folder + "/InputImages"
    files = os.listdir(path)
    files = natsort.natsorted(files)
    for i in range(len(files)):
        file = files[i]
        filepath = path + "/" + file
        prefix = file.split('.')[0]
        if os.path.isfile(filepath):
            print('********    file   ********', file)
            img1 = folder + '/InputImages/' + file
            img2 = folder + '/OutputImages/' + file
            img3 = folder + '/depthimage/' + file
            img4 = folder + '/removal-D/' + file
            starttime = datetime.datetime.now()
            run(img1,img2,img3,img4,args)
            Endtime = datetime.datetime.now()
            Time = Endtime - starttime
            print('Time', Time)
    Endtime = datetime.datetime.now()
    Time = Endtime - starttime
    print('Time', Time)