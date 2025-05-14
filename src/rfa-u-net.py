# -*- coding: utf-8 -*-
"""RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

This script implements a Vision Transformer (ViT) encoder pre-trained with RETFound
weights and an Attention U-Net decoder for segmenting the choroid in OCT images.
"""

import os
import sys
import argparse
import gdown
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from timm.layers import trunc_normal_
import models_vit
from util.pos_embed import interpolate_pos_embed


def parse_args():
    parser = argparse.ArgumentParser(description="RFA-U-Net for OCT Choroid Segmentation")
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to directory with OCT images (default: ./images)')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Path to directory with mask images (default: ./masks)')
    parser.add_argument('--weights_path', type=str, default='weights/rfa_unet_best.pth',
                        help='Path to pre-trained weights file')
    parser.add_argument('--weights_type', type=str, default='none', choices=['none','retfound','rfa-unet'],
                        help='Which weights to load')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--test_only', action='store_true',
                        help='Run inference on external data only')
    parser.add_argument('--test_image_dir', type=str, default=None,
                        help='Path to external test images')
    parser.add_argument('--test_mask_dir', type=str, default=None,
                        help='Path to external test masks')
    parser.add_argument('--pixel_size_micrometers', type=float, default=10.35,
                        help='Pixel size in μm for boundary error')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Mask binarization threshold')
    args = parser.parse_args()

    # Fallback defaults if not provided
    base_dir = os.getcwd()
    if args.image_dir is None or args.mask_dir is None:
        print("[INFO] No --image_dir or --mask_dir provided; using defaults ./images and ./masks")
        if args.image_dir is None:
            args.image_dir = os.path.join(base_dir, 'images')
        if args.mask_dir is None:
            args.mask_dir = os.path.join(base_dir, 'masks')

    return args


def download_weights(weights_path, url):
    if not os.path.exists(weights_path):
        print(f"Downloading weights to {weights_path}...")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        gdown.download(url, weights_path, quiet=False)
    else:
        print(f"Weights already at {weights_path}")


class OCTDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, num_classes=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_classes = num_classes
        self.images = sorted([f for f in os.listdir(image_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + '.tif'
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.from_numpy(np.array(mask)).long()
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.dim() == 2:
            mask = F.one_hot(mask, num_classes=self.num_classes).permute(2,0,1).float()
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        return image, mask


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(in_c[0], out_c,1), nn.BatchNorm2d(out_c))
        self.Ws = nn.Sequential(nn.Conv2d(in_c[1], out_c,1), nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(nn.Conv2d(out_c,out_c,1), nn.Sigmoid())
    def forward(self,g,s):
        g1 = self.Wg(g); s1 = self.Ws(s)
        if g1.shape[-2:]!=s1.shape[-2:]:
            s1 = F.interpolate(s1, size=g1.shape[-2:], mode='bilinear', align_corners=True)
        out = self.relu(g1+s1)
        out = self.output(out)
        if out.shape[-2:]!=s.shape[-2:]:
            out = F.interpolate(out, size=s.shape[-2:], mode='bilinear', align_corners=True)
        return out * s


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, reduce_skip=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce_x = nn.Conv2d(in_c[0], out_c,1)
        self.reduce_s = nn.Conv2d(in_c[1], out_c,1) if reduce_skip else nn.Identity()
        self.ag = AttentionGate([out_c,out_c], out_c)
        self.conv = ConvBlock(out_c*2, out_c)
    def forward(self,x,s):
        x = self.up(x); x = self.reduce_x(x)
        s = self.reduce_s(s); s = self.ag(x,s)
        x = torch.cat([x,s], dim=1)
        return self.conv(x)


class AttentionUNetViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = models_vit.RETFound_mae(
            num_classes=config['num_classes'], drop_path_rate=0.2, global_pool=True
        )
        self.encoder.patch_embed.proj = nn.Conv2d(3, config['hidden_dim'],
                                                  kernel_size=(config['patch_size'],)*2,
                                                  stride=(config['patch_size'],)*2)
        if config.get('retfound_weights_path'):
            ckpt = torch.load(config['retfound_weights_path'], map_location='cpu')
            model_sd = ckpt.get('model',ckpt)
            # remove mismatched
            sd = self.encoder.state_dict()
            for k in ['patch_embed.proj.weight','patch_embed.proj.bias','head.weight','head.bias']:
                if k in model_sd and model_sd[k].shape!=sd[k].shape:
                    del model_sd[k]
            interpolate_pos_embed(self.encoder, model_sd)
            self.encoder.load_state_dict(model_sd,strict=False)
            trunc_normal_(self.encoder.head.weight, std=2e-5)
        # decoder
        self.d1 = DecoderBlock([config['hidden_dim'],config['hidden_dim']],512)
        self.d2 = DecoderBlock([512,config['hidden_dim']],256)
        self.d3 = DecoderBlock([256,config['hidden_dim']],128)
        self.d4 = DecoderBlock([128,config['hidden_dim']],64)
        self.output = nn.Conv2d(64, config['num_classes'],1)
    def forward(self,x):
        bs = x.shape[0]
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:,1:,:]
        x = self.encoder.pos_drop(x)
        skips=[]
        for i,blk in enumerate(self.encoder.blocks):
            x = blk(x)
            if i in [5,11,17,23]: skips.append(x)
        z6,z12,z18,z24 = skips
        def to_map(z): return z.transpose(1,2).reshape(bs, z.shape[2],
                                                       int(np.sqrt(z.shape[1])),
                                                       int(np.sqrt(z.shape[1])))
        z6, z12, z18, z24 = map(to_map, (z6,z12,z18,z24))
        x = self.d1(z24, z18)
        x = self.d2(x, z12)
        x = self.d3(x, z6)
        x = self.d4(x, z6)
        return self.output(x)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__(); self.alpha, self.beta, self.smooth = alpha,beta,smooth
    def forward(self,outputs,targets):
        out = torch.sigmoid(outputs).view(-1)
        tgt = targets.view(-1)
        tp = (out*tgt).sum()
        fn = ((1-out)*tgt).sum()
        fp = (out*(1-tgt)).sum()
        t = (tp+self.smooth)/(tp+self.alpha*fn+self.beta*fp+self.smooth)
        return 1-t


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): super().__init__(); self.smooth=smooth
    def forward(self,outputs,targets):
        out = torch.sigmoid(outputs).view(-1)
        tgt=targets.view(-1)
        inter=(out*tgt).sum()
        dice=(2*inter+self.smooth)/(out.sum()+tgt.sum()+self.smooth)
        return 1-dice


def dice_score(outputs, targets, smooth=1e-6):
    out_flat = torch.sigmoid(outputs).view(-1)
    tgt_flat = targets.view(-1)
    inter = (out_flat*tgt_flat).sum()
    dice_all = (2*inter)/(out_flat.sum()+tgt_flat.sum())
    out_ch = torch.sigmoid(outputs[:,1]).view(-1)
    tgt_ch = targets[:,1].view(-1)
    dice_ch = (2*(out_ch*tgt_ch).sum()+smooth)/(out_ch.sum()+tgt_ch.sum()+smooth)
    return dice_all.item(), dice_ch.item()


def find_boundaries(mask):
    ups,low=[] ,[]
    h,w=mask.shape
    for col in range(w):
        ys=np.where(mask[:,col]>0)[0]
        ups.append(int(ys[0]) if ys.size else None)
        low.append(int(ys[-1]) if ys.size else None)
    return ups, low


def compute_errors(pred_b, gt_b, pixel_size):
    se,ue=[],[]
    for p,g in zip(pred_b,gt_b):
        if p is None or g is None: continue
        d=(p-g)*pixel_size; se.append(d); ue.append(abs(d))
    if not se: return 0.0,0.0
    return float(np.mean(se)), float(np.mean(ue))


def plot_boundaries(images, true_masks, pred_masks, threshold):
    bs=images.shape[0]
    for i in range(bs):
        img=images[i].cpu().permute(1,2,0).numpy()
        tm=true_masks[i,1].cpu().numpy()>0.5
        pm=pred_masks[i,1].cpu().numpy()>threshold
        pu,pl=find_boundaries(pm); gu,gl=find_boundaries(tm)
        us,uu=compute_errors(pu,gu, args.pixel_size_micrometers)
        ls,lu=compute_errors(pl,gl, args.pixel_size_micrometers)
        print(f"Img {i+1} U_err:{us:.2f}/{uu:.2f}, L_err:{ls:.2f}/{lu:.2f} μm")
        comb=img.copy()
        for c in range(img.shape[1]):
            if gu[c]!=None: comb[gu[c],c]=[1,0,0]
            if gl[c]!=None: comb[gl[c],c]=[0,1,0]
            if pu[c]!=None: comb[pu[c],c]=[0,0,1]
            if pl[c]!=None: comb[pl[c],c]=[1,1,0]
        plt.figure(figsize=(16,4))
        for j,(d,title) in enumerate(zip([img,tm,pm,comb],
                                         ['Image','True Mask','Pred Mask','Boundaries'])):
            plt.subplot(1,4,j+1); plt.imshow(d); plt.title(title); plt.axis('off')
        plt.show()


def train_fold(train_loader, valid_loader, test_loader,
               model, criterion, optimizer, device,
               num_epochs, scaler, threshold):
    model.train()
    for e in range(num_epochs):
        epoch_loss=0.0
        for imgs,msks in train_loader:
            imgs,msks=imgs.to(device),msks.to(device)
            optimizer.zero_grad()
            with autocast(device.type):
                outs=model(imgs)
                loss=criterion(outs,msks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            epoch_loss+=loss.item()
        print(f"Epoch {e+1}/{num_epochs} Loss:{epoch_loss/len(train_loader):.4f}")
    model.eval()
    dc, dch=[] ,[]
    upe, upu, lse, lpu=[],[],[],[]
    with torch.no_grad():
        for imgs,msks in valid_loader:
            imgs,msks=imgs.to(device),msks.to(device)
            outs=model(imgs)
            da,db=dice_score(outs,msks)
            dc.append(da); dch.append(db)
            pms=torch.sigmoid(outs).cpu().numpy(); tms=msks.cpu().numpy()
            for i in range(imgs.size(0)):
                pm,pgt = pms[i,1]>threshold, tms[i,1]>0.5
                pu,pl=find_boundaries(pm); gu,gl=find_boundaries(pgt)
                us,uu=compute_errors(pu,gu,args.pixel_size_micrometers)
                ls,lu=compute_errors(pl,gl,args.pixel_size_micrometers)
                upe.append(us); upu.append(uu)
                lse.append(ls); lpu.append(lu)
    # visualize test first batch
    with torch.no_grad():
        for imgs,msks in test_loader:
            imgs,msks=imgs.to(device),msks.to(device)
            outs=model(imgs)
            plot_boundaries(imgs,msks,torch.sigmoid(outs),threshold)
            break
    return np.mean(dc), np.mean(dch), np.mean(upe), np.mean(upu), np.mean(lse), np.mean(lpu)


if __name__ == '__main__':
    args = parse_args()
    # Set up transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(size=(args.image_size, args.image_size), scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights
    RETFOUND_PATH='weights/RETFound_oct_weights.pth'
    RFA_PATH='weights/rfa_unet_best.pth'
    RFA_URL='https://drive.google.com/uc?id=1q2giAcI8ASe2qnA9L69Mqb01l2qKjTV0'
    if args.weights_type=='retfound':
        download_weights(RETFOUND_PATH,None)
        wpath=RETFOUND_PATH
    elif args.weights_type=='rfa-unet':
        download_weights(RFA_PATH,RFA_URL)
        wpath=RFA_PATH
    else: wpath=None
    config={'image_size':args.image_size,'hidden_dim':1024,'patch_size':16,
            'num_classes':2,'retfound_weights_path':wpath}
    model=AttentionUNetViT(config).to(device)
    if wpath:
        ckpt=torch.load(wpath,map_location=device)
        model.load_state_dict(ckpt,strict=False)
    if args.test_only:
        assert args.test_image_dir and args.test_mask_dir, "--test_only requires --test_image_dir and --test_mask_dir"
        test_ds = OCTDataset(
            args.test_image_dir,
            args.test_mask_dir,
            transform=val_transform,
            num_classes=2
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        all_d=[]; ue=[]; le=[]
        with torch.no_grad():
            for imgs,msks in test_loader:
                imgs,msks=imgs.to(device),msks.to(device)
                outs=model(imgs)
                _,db=dice_score(outs,msks)
                all_d.append(db)
                pns=torch.sigmoid(outs).cpu().numpy(); tns=msks.cpu().numpy()
                for i in range(imgs.size(0)):
                    pm= pns[i,1]>args.threshold; tm=tns[i,1]>0.5
                    pu,pl=find_boundaries(pm); gu,gl=find_boundaries(tm)
                    us,uu=compute_errors(pu,gu,args.pixel_size_micrometers)
                    ls,lu=compute_errors(pl,gl,args.pixel_size_micrometers)
                    ue.append((us,uu)); le.append((ls,lu))
        print(f"Choroid Dice: {np.mean(all_d):.4f}")
        usm,uum=np.mean([x for x,_ in ue]),np.mean([y for _,y in ue])
        lsm,lum=np.mean([x for x,_ in le]),np.mean([y for _,y in le])
        print(f"Upper error: {usm:.2f}/{uum:.2f} μm  Lower error: {lsm:.2f}/{lum:.2f} μm")
        sys.exit(0)
    # split data
    full_ds=OCTDataset(args.image_dir,args.mask_dir,
                       image_transform=val_transform,
                       mask_size=(args.image_size,args.image_size))
    n=len(full_ds)
    t=int(0.7*n); v=int(0.15*n)
    train_ds,valid_ds,test_ds=random_split(full_ds,[t,v,n-t-v])
    train_ds.dataset.image_transform=train_transform
    valid_ds.dataset.image_transform=val_transform
    test_ds.dataset.image_transform=val_transform
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True)
    valid_loader=DataLoader(valid_ds,batch_size=args.batch_size,shuffle=False,num_workers=2,pin_memory=True)
    test_loader =DataLoader(test_ds, batch_size=args.batch_size,shuffle=False,num_workers=2,pin_memory=True)
    # freeze encoder except last block
    for i,blk in enumerate(model.encoder.blocks):
        if i<23:
            for p in blk.parameters(): p.requires_grad=False
    criterion=TverskyLoss(alpha=0.7,beta=0.3).to(device)
    optimizer=optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=1e-4)
    scaler=GradScaler()
    results=train_fold(train_loader,valid_loader,test_loader,
                       model,criterion,optimizer,device,
                       args.num_epochs,scaler,args.threshold)
    print(f"Results: Combined Dice={results[0]:.4f}, Choroid Dice={results[1]:.4f},"
          f" Upper Err={results[2]:.2f}/{results[3]:.2f},"
          f" Lower Err={results[4]:.2f}/{results[5]:.2f}")
