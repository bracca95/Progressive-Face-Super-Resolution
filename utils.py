import torch.nn.functional as F
from face_alignment.utils import *

def inf_train_gen(dataloader):
    while True:
        for _, data in enumerate(dataloader):
            yield data

def get_heat_map(face_alignment_net, fake_image, real_image, Fan=False, scale_factor=None):
    if Fan:#use fan model
        hm_f = face_alignment_net(F.interpolate(0.5*fake_image+0.5, scale_factor=scale_factor, mode='bilinear'))[-1]
        hm_r = face_alignment_net(F.interpolate(0.5*real_image+0.5, scale_factor=scale_factor, mode='bilinear'))[-1]
    else:#use our student model
        hm_f = face_alignment_net(0.5*fake_image+0.5)
        hm_r = face_alignment_net(0.5*real_image+0.5)
    pts_r, pts_img_r = get_preds_fromhm(hm_r)
    pts_r, pts_img_r = pts_r.view(-1, 68, 2), pts_img_r.view(-1, 68, 2)
    #heatmap is normalized
    HMax = torch.max(torch.max(torch.max(hm_r, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0], dim=1,keepdim=True)[0]
    HMin = torch.min(torch.min(torch.min(hm_r, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0], dim=1,keepdim=True)[0]
    hm_r = (hm_r - HMin)/(HMax-HMin)

    HMax = torch.max(torch.max(torch.max(hm_f, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0], dim=1,keepdim=True)[0]
    Hmin = torch.min(torch.min(torch.min(hm_f, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0], dim=1,keepdim=True)[0]
    hm_f = (hm_f - HMin)/(HMax-HMin)
    return hm_f, hm_r

def operate_heatmap(hm):
    h_max = torch.max(hm)
    h_min = torch.min(hm)
    heatMap = torch.max(hm, dim=1)[0]
    heatMap = (heatMap - h_min) / (h_max - h_min)
    b, c, x, y = hm.shape
    heatMap = heatMap.view(b, 1, x, y)
    return heatMap

def requires_grad(model, value=True):
    for param in model.parameters():
        param.requires_grad = value