#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from kornia.filters import spatial_gradient
from .image_utils import psnr
from utils.image_utils import erode
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()



# utils/loss_utils.py 파일 내

def calculate_loss(viewpoint_camera, pc, render_pkg, opt, iteration):
    
    # 1. 렌더링된 가우시안이 아예 없는 경우 (가장 먼저 검사)
    #    (KeyError 방지를 위해 모든 키를 포함하여 반환)
    if render_pkg["visibility_filter"].sum() == 0:
        image = render_pkg["render"]
        gt_image = viewpoint_camera.original_image.cuda()
        loss = l1_loss(image, gt_image)
        
        tb_dict = {
            "loss_dist": 0.0, 
            "loss_normal_render_depth": 0.0, 
            "loss0": loss.item(), 
            "loss_l1": loss.item(), 
            "loss_normal_smooth": 0.0, 
            "loss_depth_smooth": 0.0,
            "loss": loss.item(), 
            "psnr": 0.0, 
            "ssim": 0.0,
            "num_points": pc.get_xyz.shape[0]
        }
        return loss, tb_dict

    # --- 렌더링 결과물 언패킹 ---
    tb_dict = { "num_points": pc.get_xyz.shape[0] }
    rendered_image = render_pkg["render"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"]
    rend_dist = render_pkg["rend_dist"]
    gt_image = viewpoint_camera.original_image.cuda()
    uncertainty_map = render_pkg.get("rgb_uncertainty_map")

    # 2. 학습 단계에 따라 주 손실(image loss)을 동적으로 선택
    # 불확실성 맵이 비어있지 않은지 한번 더 검사
    if opt.uncertainty_from_iter > 0 and iteration > opt.uncertainty_from_iter and uncertainty_map is not None and uncertainty_map.numel() > 0:
        # 하이브리드 손실 (L1 + NLL) 계산
        uncertainty_map = uncertainty_map + 1e-8
        l2_error = (rendered_image - gt_image) ** 2
        nll_loss_map = 0.5 * (l2_error / uncertainty_map**2 + torch.log(uncertainty_map**2))
        #print("ucmap",uncertainty_map.shape)
        Lnll = nll_loss_map.mean()
        Ll1 = l1_loss(rendered_image, gt_image)
        image_loss = (1.0 - opt.lambda_hybrid) * Ll1 + opt.lambda_hybrid * Lnll
        tb_dict["loss_nll"] = Lnll.item()
        tb_dict["loss_l1"] = Ll1.item()
    else:
        # 일반 학습 단계에서는 기존 L1 손실만 사용
        image_loss = l1_loss(rendered_image, gt_image)
        tb_dict["loss_l1"] = image_loss.item()
    
    # --- 3. 주 손실과 SSIM 손실 조합 ---
    ssim_val = ssim(rendered_image.unsqueeze(0), gt_image.unsqueeze(0))
    loss0 = (1.0 - opt.lambda_dssim) * image_loss + opt.lambda_dssim * (1.0 - ssim_val)
    loss = loss0.clone()

    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()

    # --- 4. 나머지 정규화(Regularization) 손실들 추가 ---
    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None].mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss += opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = 0.0

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss.item()
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = 0.0

    if opt.lambda_normal_smooth > 0 and iteration > opt.normal_smooth_from_iter and iteration < opt.normal_smooth_until_iter:
        loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        loss += opt.lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = 0.0
    
    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        loss += opt.lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = 0.0

    
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict