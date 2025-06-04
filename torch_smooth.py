
import torch
print("Torch sees these devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
print("Current device index:", torch.cuda.current_device())
print("Selected device:", torch.cuda.get_device_name(torch.cuda.current_device()))



import argparse, glob, os, random
from typing import List, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassF1Score
from torchvision import transforms as T
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

from lightly.loss import NTXentLoss
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.models.utils import deactivate_requires_grad, activate_requires_grad
try:                                  # lightly ? 1.5
    from lightly.optim import LARS
except ImportError:                   # lightly 1.0-1.4
    try:
        from lightly.optim.lars import LARS
    except ImportError:
        LARS = None                  # will fall back to SGD

# ??? CLI ------------------------------------------------------------------
P = argparse.ArgumentParser()
P.add_argument("--root",          default=".")
P.add_argument("--stage",         choices=["ssl", "finetune", "all"],
               default="all")
P.add_argument("--batch_ssl",     type=int, default=256)
P.add_argument("--batch_sup",     type=int, default=160)
P.add_argument("--epochs_ssl",    type=int, default=200)
P.add_argument("--epochs_sup",    type=int, default=50)
P.add_argument("--device",        default="cuda")
P.add_argument("--img_small",     type=int, default=224)
P.add_argument("--img_big",       type=int, default=384)
P.add_argument("--stride_small",  type=int, default=160)
P.add_argument("--stride_big",    type=int, default=256)
P.add_argument("--folds",         type=int, default=5)
args = P.parse_args()

# ??? environment ----------------------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV  = torch.device("cuda:0")
AMP  = DEV.type == "cuda"
os.makedirs("checkpoints", exist_ok=True)

# ??? helper: exhaustive tiling -------------------------------------------
def tile_crop(img: Image.Image, size: int, stride: int) -> List[Image.Image]:
    w, h = img.size; out=[]
    for top in range(0, h - size + 1, stride):
        for left in range(0, w - size + 1, stride):
            out.append(img.crop((left, top, left+size, top+size)))
    if (h-size)%stride:
        for left in range(0, w - size + 1, stride):
            out.append(img.crop((left, h-size, left+size, h)))
    if (w-size)%stride:
        for top in range(0, h - size + 1, stride):
            out.append(img.crop((w-size, top, w, top+size)))
    if (h-size)%stride and (w-size)%stride:
        out.append(img.crop((w-size, h-size, w, h)))
    return out

# ??? data discovery -------------------------------------------------------
def discover(root: str) -> Tuple[List[str], List[int], List[float]]:
    imgs,lbls,depths = [],[],[]
    for mat in sorted(glob.glob(os.path.join(root, "material_*"))):
        tail = mat.split("_")[-1]
        if not tail.isdigit(): continue
        y = int(tail) - 1
        for fr in glob.glob(os.path.join(mat, "cycle_*", "frame.png")):
            imgs.append(fr); lbls.append(y); depths.append(1.0)
    return imgs,lbls,depths

IMGS, LABELS, DEPTHS = discover(args.root)
NCLS = len(set(LABELS))
print(f"Loaded {len(IMGS)} frames across {NCLS} classes")

# ??? transforms -----------------------------------------------------------
MEAN,STD = IMAGENET_NORMALIZE["mean"], IMAGENET_NORMALIZE["std"]
def aug(size):
    return T.Compose([
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(20), T.ColorJitter(0.3,0.3,0.3,0.1),
        T.ToTensor(), T.Normalize(MEAN,STD)])
def val(size):
    return T.Compose([T.ToTensor(), T.Normalize(MEAN,STD)])

A_SM,A_BG = aug(args.img_small), aug(args.img_big)
V_SM,V_BG = val(args.img_small), val(args.img_big)

# ??? datasets -------------------------------------------------------------
class SSLTiles(Dataset):
    """Return two independent augmented views of the same raw tile."""
    def __init__(self, paths, size, stride):
        self.tiles = []
        for p in paths:
            self.tiles += tile_crop(Image.open(p).convert("RGB"), size, stride)
        self.tf = aug(size)
    def __len__(self): return len(self.tiles)
    def __getitem__(self, idx):
        t = self.tiles[idx]
        return self.tf(t), self.tf(t)

class SupTiles(Dataset):
    def __init__(self, paths, lbls, depths, train=True):
        self.records = list(zip(paths, lbls, depths)); self.train=train
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        p,l,d = self.records[idx]
        img = Image.open(p).convert("RGB")
        sm_tiles = tile_crop(img, args.img_small, args.stride_small)
        bg_tiles = tile_crop(img, args.img_big,   args.stride_big)
        tf_sm = A_SM if self.train else V_SM
        tf_bg = A_BG if self.train else V_BG
        sm = torch.stack([tf_sm(t) for t in random.sample(
                          sm_tiles, min(6,len(sm_tiles)))])
        bg = torch.stack([tf_bg(t) for t in random.sample(
                          bg_tiles, min(4,len(bg_tiles)))])
        depth_code = torch.zeros(2)          # dummy depth
        return sm,bg,l,depth_code

# ??? model ----------------------------------------------------------------
class DualPyramid(nn.Module):
    def __init__(self, nc:int):
        super().__init__()
        self.fe = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
        in_dim  = self.fe.classifier[1].in_features
        self.fe.classifier = nn.Identity()
        self.gap, self.gmp = nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_dim*2+2, 1024), nn.BatchNorm1d(1024), nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(1024,512), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, nc))
    def enc(self,x):                         # (T,3,H,W)
        f = self.fe(x)
        return torch.cat([self.gap(f.T), self.gmp(f.T)],1).squeeze(-1)
    def forward(self,sm,bg,depth):
        z = torch.cat([self.enc(sm).mean(0,keepdim=True),
                       self.enc(bg).mean(0,keepdim=True),
                       depth],1)
        return self.head(z)

# ??? SimCLR pre-train -----------------------------------------------------
def simclr_pretrain(backbone, dset):
    dl = DataLoader(dset, batch_size=args.batch_ssl, shuffle=True,
                    num_workers=8, drop_last=True, pin_memory=True)
    deactivate_requires_grad(backbone)

    proj = nn.Sequential(
        nn.Linear(2304,1024), nn.BatchNorm1d(1024), nn.ReLU(),
        nn.Linear(1024,256)).to(DEV)
    base_opt = torch.optim.SGD(proj.parameters(), lr=1e-3,
                               momentum=0.9, weight_decay=1e-6)
    opt = LARS(base_opt) if LARS else base_opt
    loss_fn = NTXentLoss().to(DEV)

    for ep in range(1, args.epochs_ssl+1):
        for v1,v2 in tqdm(dl, desc=f"SSL {ep}/{args.epochs_ssl}"):
            v1,v2 = v1.to(DEV), v2.to(DEV)
            with torch.no_grad(), torch.autocast(device_type=DEV.type,
                                                 enabled=AMP):
           
                 h1, h2 = backbone.fe(v1), backbone.fe(v2)
     
            z1, z2 = proj(h1.float()), proj(h2.float()) 
            loss = loss_fn(z1,z2)
            opt.zero_grad(); loss.backward(); opt.step()
    activate_requires_grad(backbone)
    torch.save(backbone.state_dict(), "checkpoints/ssl_b6_rgb.pt")

# ??? supervised phase -----------------------------------------------------
def sup_train(model,train_ds,val_ds):
    dl_tr=DataLoader(train_ds,batch_size=1,shuffle=True,
                     num_workers=6,pin_memory=True)
    dl_va=DataLoader(val_ds,batch_size=1,shuffle=False,
                     num_workers=4,pin_memory=True)
    opt=optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sched=optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-4,
                 steps_per_epoch=len(dl_tr),epochs=args.epochs_sup)
    ce_ls=lambda x,y: nn.functional.cross_entropy(x,y,label_smoothing=0.05)
    f1=MulticlassF1Score(NCLS,average="macro").to(DEV)
    best,best_sd=0,None

    for ep in range(1,args.epochs_sup+1):
        model.train()
        for sm,bg,l,code in tqdm(dl_tr,desc=f"CE Ep{ep}"):
            sm,bg=sm[0].to(DEV),bg[0].to(DEV)
            l=torch.tensor([l],device=DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEV.type,enabled=AMP):
                loss=ce_ls(model(sm,bg,code.to(DEV)),l)
            loss.backward(); opt.step(); sched.step()

        model.eval(); vp,gt=[],[]
        with torch.no_grad(), torch.autocast(device_type=DEV.type,enabled=AMP):
            for sm,bg,l,code in dl_va:
                sm,bg=sm[0].to(DEV),bg[0].to(DEV)
                vp.append(model(sm,bg,code.to(DEV)).softmax(1))
                gt.append(torch.tensor([l],device=DEV))
        sc=f1(torch.cat(vp),torch.cat(gt)).item()
        print(f"  val F1={sc:.3f}")
        if sc>best: best,best_sd=sc,model.state_dict().copy()
    return best_sd

# ??? K-fold orchestration --------------------------------------------------
skf=StratifiedKFold(args.folds,shuffle=True,random_state=42)
ALLP,ALLT=[],[]
for k,(tr,va) in enumerate(skf.split(IMGS,LABELS),1):
    print(f"\n== Fold {k}/{args.folds} ==")
    tr_i=[IMGS[i] for i in tr]; va_i=[IMGS[i] for i in va]
    tr_l=[LABELS[i] for i in tr]; va_l=[LABELS[i] for i in va]
    tr_d=[DEPTHS[i] for i in tr]; va_d=[DEPTHS[i] for i in va]

    net=DualPyramid(NCLS).to(DEV)
    if args.stage in ("ssl","all"):
        simclr_pretrain(net, SSLTiles(tr_i,args.img_small,args.stride_small))
    if args.stage in ("finetune","all"):
        best= sup_train(net,
                       SupTiles(tr_i,tr_l,tr_d,train=True),
                       SupTiles(va_i,va_l,va_d,train=False))
        net.load_state_dict(best)
    torch.save(net.state_dict(),f"checkpoints/fold{k}.pt")

    net.eval(); pr,gt=[],[]
    with torch.no_grad(), torch.autocast(device_type=DEV.type,enabled=AMP):
        for sm,bg,l,code in DataLoader(
            SupTiles(va_i,va_l,va_d,train=False),
            batch_size=1,shuffle=False,pin_memory=True):
            sm,bg=sm[0].to(DEV),bg[0].to(DEV)
            pr.append(net(sm,bg,code.to(DEV)).softmax(1))
            gt.append(torch.tensor([l],device=DEV))
    ALLP.append(torch.cat(pr)); ALLT.append(torch.cat(gt))

# ??? final report ----------------------------------------------------------
print("\n===== 5-fold ensemble =====")
PRED=torch.cat(ALLP).argmax(1).cpu().numpy()
TRUE=torch.cat(ALLT).cpu().numpy()
print(classification_report(TRUE,PRED,digits=4))
print(confusion_matrix(TRUE,PRED))
