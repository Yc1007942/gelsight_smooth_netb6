#!/usr/bin/env python3
# ============================================================================
#   GelSight smooth-texture classifier – “big-GPU” edition
#   usage:  CUDA_VISIBLE_DEVICES=0 python gelsight_big.py --stage all
# ============================================================================

import argparse, glob, os, random, math, json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassF1Score
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
from lightly.loss import NTXentLoss
from lightly.models.utils import deactivate_requires_grad, activate_requires_grad
try:
    from lightly.optim import LARS
except Exception:
    try: from lightly.optim.lars import LARS
    except Exception: LARS = None

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

# ------------------------- CLI -------------------------------------------
P = argparse.ArgumentParser()
P.add_argument("--root", default=".")
P.add_argument("--stage", choices=["ssl","finetune","all"], default="all")
P.add_argument("--batch_ssl", type=int, default=256)
P.add_argument("--batch_sup", type=int, default=512)
P.add_argument("--epochs_ssl", type=int, default=200)
P.add_argument("--epochs_sup", type=int, default=60)
P.add_argument("--folds",      type=int, default=5)
P.add_argument("--device",     default="cuda")
args = P.parse_args()

# ------------------------- misc setup ------------------------------------
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV=torch.device(args.device if torch.cuda.is_available() else "cpu")
AMP= DEV.type=="cuda"
os.makedirs("checkpoints",exist_ok=True)

IMG_S, IMG_B = 224, 384
STRIDE_S, STRIDE_B = 112, 192

# ------------------------- helper: random tile ---------------------------
def random_tile(img:Image.Image, size:int, stride:int):
    w,h = img.size
    x = random.randrange(0, w-size+1, stride)
    y = random.randrange(0, h-size+1, stride)
    return img.crop((x,y,x+size,y+size))

# ------------------------- heavy aug -------------------------------------
MEAN,STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
def _albumentations_heavy(sz:int)->A.Compose:
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.88,1.12), rotate=(-12,12),
                 translate_percent=0.07, shear=(-8,8), p=0.7),
        A.HorizontalFlip(0.5),  A.VerticalFlip(0.3),
        A.RandomBrightnessContrast(0.25,0.25,0.4),
        A.HueSaturationValue(8,12,8,0.4),
        A.ImageCompression(60,100,0.3),
        A.GaussNoise(10,40,0.4),
        A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2,
                       shadow_dimension=5, p=0.25),
        A.CoarseDropout(max_holes=6, max_height=int(0.12*sz),
                        max_width=int(0.12*sz), fill_value=None, p=0.4),
        A.Resize(sz,sz,cv2.INTER_AREA), ToTensorV2()
    ])

A_TSM = _albumentations_heavy(IMG_S)
A_TBG = _albumentations_heavy(IMG_B)
V_TFM = A.Compose([A.Resize(IMG_S,IMG_S), ToTensorV2()])

# ------------------------- discovery -------------------------------------
def discover(root)->Tuple[List[str],List[int],List[float]]:
    img,lbl,depth = [],[],[]
    for m in sorted(glob.glob(os.path.join(root,"material_*"))):
        y=int(m.split("_")[-1])-1
        for cyc in glob.glob(os.path.join(m,"cycle_*")):
            meta = Path(cyc)/"meta.json"
            d_mm = json.load(open(meta))["depth_mm"] if meta.exists() else 1.0
            for fr in glob.glob(os.path.join(cyc,"frame*.png")):
                img.append(fr); lbl.append(y); depth.append(d_mm)
    return img,lbl,depth

IMGS,LABELS,DEPTHS = discover(args.root)
NCLS=len(set(LABELS))
print(f"✔ {len(IMGS)} frames  |  {NCLS} classes")

# ------------------------- datasets --------------------------------------
class SSLTiles(Dataset):
    def __init__(self, paths):
        self.paths=paths
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        img=Image.open(self.paths[i]).convert("RGB")
        return (A_TSM(image=np.array(random_tile(img,IMG_S,STRIDE_S)))['image'],
                A_TSM(image=np.array(random_tile(img,IMG_S,STRIDE_S)))['image'])

class SupTiles(Dataset):
    def __init__(self, paths,lbls,depths,train=True):
        self.recs=list(zip(paths,lbls,depths)); self.train=train
    def __len__(self): return len(self.recs)
    def __getitem__(self,i):
        p,l,d=self.recs[i]
        im=Image.open(p).convert("RGB")
        sm=torch.stack([A_TSM(image=np.array(random_tile(im,IMG_S,STRIDE_S)))['image']
                        for _ in range(6)])
        bg=torch.stack([A_TBG(image=np.array(random_tile(im,IMG_B,STRIDE_B)))['image']
                        for _ in range(4)])
        depth_code=torch.tensor([math.sin(d/5.0), math.cos(d/5.0)],
                                dtype=torch.float32)
        return sm,bg,l,depth_code

# -------------------- model ----------------------------------------------
class LocalGlobal(nn.Module):
    def __init__(self, dim=2304, heads=4):
        super().__init__()
        self.local = nn.MultiheadAttention(dim,heads,batch_first=True)
        self.glob  = nn.MultiheadAttention(dim,heads,batch_first=True)
        self.proj  = nn.Linear(dim,dim)
    def forward(self, x):                 # x: (T,D)
        loc,_ = self.local(x.unsqueeze(0),x.unsqueeze(0),x.unsqueeze(0))
        glo,_ = self.glob(loc,loc,loc)
        return self.proj(glo.mean(1)).squeeze(0)  # (D,)

class DualPyramid(nn.Module):
    def __init__(self,nc):
        super().__init__()
        self.fe = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
        for name,p in self.fe.named_parameters():
            layer=int(name.split('.')[0].lstrip('_blocks') or 0)
            p.requires_grad = layer>=3            # freeze first 3 stages
        d = self.fe.classifier[1].in_features
        self.fe.classifier = nn.Identity()
        self.attn = LocalGlobal(d)
        self.head = nn.Sequential(
            nn.Linear(d*2+2,1024), nn.BatchNorm1d(1024), nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(1024,512), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512,nc))
    def _agg(self,t):                           # t: (B,T,3,H,W)
        b,t = t.shape[:2]
        f = self.fe(t.flatten(0,1))             # (B*T,D)
        f = f.view(b,-1,f.size(-1))             # (B,T,D)
        return self.attn(f)                     # (B,D)
    def forward(self,sm,bg,depth):
        z=torch.cat([self._agg(sm), self._agg(bg), depth],1)
        return self.head(z)

# -------------------- MixUp / CutMix -------------------------------------
def mixup_cutmix(x,y,alpha=0.4):
    lam=np.random.beta(alpha,alpha)
    idx=torch.randperm(x.size(0),device=x.device)
    if random.random()<0.5:        # MixUp
        x_mix = lam*x + (1-lam)*x[idx]
    else:                          # CutMix patch on all tiles
        b,c,h,w = x.shape
        cx,cy = np.random.randint(w), np.random.randint(h)
        cut=int(h*math.sqrt(1-lam))
        x_mix=x.clone()
        x_mix[:,:,cy:cy+cut,cx:cx+cut]=x[idx,:,cy:cy+cut,cx:cx+cut]
    y_mix = lam*y + (1-lam)*y[idx]
    return x_mix,y_mix

# -------------------- SSL pre-train --------------------------------------
def ssl_pretrain(backbone,paths):
    dl=DataLoader(SSLTiles(paths),batch_size=args.batch_ssl,
                  shuffle=True,num_workers=8,drop_last=True,pin_memory=True)
    proj=nn.Sequential(nn.Linear(2304,1024),nn.BatchNorm1d(1024),nn.ReLU(),
                       nn.Linear(1024,256)).to(DEV)
    opt_base=optim.SGD(proj.parameters(),lr=1e-3,momentum=0.9,weight_decay=1e-6)
    opt=LARS(opt_base) if LARS else opt_base
    ntx=NTXentLoss().to(DEV)
    activate_requires_grad(backbone)
    for ep in range(1,args.epochs_ssl+1):
        for v1,v2 in tqdm(dl,desc=f"SSL {ep}/{args.epochs_ssl}"):
            v1,v2=v1.to(DEV),v2.to(DEV)
            with torch.autocast(device_type=DEV.type,enabled=AMP):
                h1,h2=backbone.fe(v1),backbone.fe(v2)
            loss=ntx(proj(h1.float()),proj(h2.float()))
            opt.zero_grad(); loss.backward(); opt.step()
    activate_requires_grad(backbone)

# -------------------- supervised train -----------------------------------
def train_one_fold(net,paths,lbls,depths):
    tr_ds=SupTiles(paths,lbls,depths,True)
    va_ds=SupTiles(paths,lbls,depths,False)    # same split object outside
    dl_tr=DataLoader(tr_ds,batch_size=args.batch_sup,shuffle=True,
                     num_workers=8,pin_memory=True)
    dl_va=DataLoader(va_ds,batch_size=args.batch_sup,shuffle=False,
                     num_workers=4,pin_memory=True)
    opt=optim.AdamW([p for p in net.parameters() if p.requires_grad],
                    lr=3e-4,weight_decay=1e-4)
    sched=optim.lr_scheduler.OneCycleLR(opt,3e-4,len(dl_tr),args.epochs_sup)
    f1=MulticlassF1Score(NCLS,average='macro').to(DEV)
    ema=torch.optim.swa_utils.AveragedModel(net)
    best=0; best_state=None

    for ep in range(1,args.epochs_sup+1):
        net.train()
        for sm,bg,l,code in tqdm(dl_tr,desc=f"Ep{ep}/{args.epochs_sup}"):
            sm,bg=sm.to(DEV),bg.to(DEV)
            y=nn.functional.one_hot(l.to(DEV),NCLS).float()
            sm,bg = mixup_cutmix(sm,bg) if random.random()<0.5 else (sm,bg)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEV.type,enabled=AMP):
                out=net(sm,bg,code.to(DEV))
                loss=(-(y*out.log_softmax(1)).sum(1)).mean()
            loss.backward(); opt.step(); sched.step(); ema.update_parameters(net)

        net.eval(); ema.eval(); vp,gt=[],[]
        with torch.no_grad(), torch.autocast(device_type=DEV.type,enabled=AMP):
            for sm,bg,l,code in dl_va:
                sm,bg=sm.to(DEV),bg.to(DEV)
                vp.append(ema(sm,bg,code.to(DEV)).softmax(1))
                gt.append(l.to(DEV))
        score=f1(torch.cat(vp),torch.cat(gt)).item()
        if score>best: best,best_state=score,ema.module.state_dict().copy()
        print(f"  val-F1={score:.4f}  best={best:.4f}")
    return best_state

# -------------------- K-fold CV ------------------------------------------
skf=StratifiedKFold(args.folds,shuffle=True,random_state=SEED)
ALLP,ALLT=[],[]
for fold,(tr,va) in enumerate(skf.split(IMGS,LABELS),1):
    print(f"\n── fold {fold}/{args.folds} ──")
    tr_i=[IMGS[i] for i in tr]; va_i=[IMGS[i] for i in va]
    tr_l=[LABELS[i] for i in tr]; va_l=[LABELS[i] for i in va]
    tr_d=[DEPTHS[i] for i in tr]; va_d=[DEPTHS[i] for i in va]

    mdl=DualPyramid(NCLS).to(DEV)
    if args.stage in ("ssl","all"):
        ssl_pretrain(mdl, tr_i)
    if args.stage in ("finetune","all"):
        best=train_one_fold(mdl,tr_i,tr_l,tr_d)
        mdl.load_state_dict(best)
    torch.save(mdl.state_dict(),f"checkpoints/fold{fold}.pt")

    # evaluate
    mdl.eval(); pr,gt=[],[]
    va_loader=DataLoader(SupTiles(va_i,va_l,va_d,False),
                         batch_size=args.batch_sup,shuffle=False,num_workers=4)
    with torch.no_grad(), torch.autocast(device_type=DEV.type,enabled=AMP):
        for sm,bg,l,code in va_loader:
            sm,bg=sm.to(DEV),bg.to(DEV)
            pr.append(mdl(sm,bg,code.to(DEV)).softmax(1)); gt.append(l.to(DEV))
    ALLP.append(torch.cat(pr)); ALLT.append(torch.cat(gt))

# -------------------- report ---------------------------------------------
PRED=torch.cat(ALLP).argmax(1).cpu().numpy()
TRUE=torch.cat(ALLT).cpu().numpy()
print("\n=== 5-fold ensemble ===")
print(classification_report(TRUE,PRED,digits=4))
print(confusion_matrix(TRUE,PRED))
