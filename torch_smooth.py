#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  High-res GelSight classifier  (global Swin-V2  +  local EfficientNet-V2)
#  - MoCo-v2 self-supervision (optional)          - 5-fold CV
#  Author: 2025-06 - Adapted for large VRAM GPUs
# ============================================================================

import argparse, glob, math, os, random
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassF1Score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm
import timm                                        # ▶ new backbone zoo
from timm.data import create_transform

# ───────────── CLI ────────────────────────────────────────────────────────
P = argparse.ArgumentParser()
P.add_argument('--root', default='.')
P.add_argument('--stage', choices=['ssl', 'finetune', 'all'], default='all')
P.add_argument('--batch_ssl', type=int, default=512)
P.add_argument('--batch_sup', type=int, default=128)        # logical batch
P.add_argument('--accum',     type=int, default=2)          # grad-acc steps
P.add_argument('--epochs_ssl', type=int, default=300)
P.add_argument('--epochs_sup', type=int, default=60)
P.add_argument('--folds',      type=int, default=5)
P.add_argument('--device',     default='cuda:0')
args = P.parse_args()

# ───────────── env / reproducibility ──────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV  = torch.device(args.device if torch.cuda.is_available() else 'cpu')
AMP  = DEV.type == 'cuda'
os.makedirs('checkpoints', exist_ok=True)

# ───────────── dataset discovery ─────────────────────────────────────────
def discover(root: str) -> Tuple[List[str], List[int]]:
    imgs, lbls = [], []
    for mat in sorted(glob.glob(os.path.join(root, 'material_*'))):
        t = mat.rsplit('_',1)[-1]
        if not t.isdigit(): continue
        y = int(t) - 1
        imgs += glob.glob(os.path.join(mat, 'cycle_*', 'frame.png'))
        lbls += [y]*len(imgs[len(lbls):])
    return imgs, lbls

IMGS, LABELS = discover(args.root)
NCLS = len(set(LABELS))
print(f'{len(IMGS)} frames  |  {NCLS} classes')

# ───────────── transforms (timm helpers) ─────────────────────────────────
global_tf_train = create_transform(
    input_size=(3,640,480), is_training=True, auto_augment='rand-m9-mstd0.5-inc1',
    interpolation='bicubic', re_prob=0.25, re_mode='pixel', re_count=1)
global_tf_val   = create_transform(
    input_size=(3,640,480), is_training=False, interpolation='bicubic')

local_tf_train  = create_transform(
    input_size=224, is_training=True, auto_augment='rand-m9-mstd0.5-inc1',
    interpolation='bicubic', re_prob=0.25)
local_tf_val    = create_transform(
    input_size=224, is_training=False, interpolation='bicubic')

# ───────────── datasets ──────────────────────────────────────────────────
class SSLDataset(Dataset):
    """Return two random views (RandAug) of one *full* image."""
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return global_tf_train(img), global_tf_train(img)

class SupDataset(Dataset):
    def __init__(self, paths, labels, train=True):
        self.recs  = list(zip(paths, labels))
        self.train = train
    def __len__(self): return len(self.recs)

    def _rand_tiles(self, img, n=6, size=224, stride=160):
        w,h = img.size; xs,ys = [],[]
        for _ in range(n*2):                          # oversample → unique
            xs.append(random.randrange(0, w-size+1, stride))
            ys.append(random.randrange(0, h-size+1, stride))
        picks = random.sample(range(len(xs)), n)
        return [img.crop((xs[i], ys[i], xs[i]+size, ys[i]+size)) for i in picks]

    def __getitem__(self, idx):
        p,l = self.recs[idx]
        img = Image.open(p).convert('RGB')
        global_img = global_tf_train(img) if self.train else global_tf_val(img)
        tiles = self._rand_tiles(img) if self.train else \
                [img.crop((0,0,224,224))]*6
        local_imgs = torch.stack([local_tf_train(t) if self.train
                                  else local_tf_val(t) for t in tiles])
        depth_code = torch.zeros(2)                  # placeholder
        return global_img, local_imgs, l, depth_code

# ───────────── model ─────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        # global branch
        self.g_back = timm.create_model(
            'swinv2_base_window12to16_224to384_22kft1k', pretrained=True)
        self.g_back.reset_classifier(0); g_dim = self.g_back.num_features
        # local branch
        self.l_back = timm.create_model('efficientnetv2_s', pretrained=True)
        self.l_back.reset_classifier(0); l_dim = self.l_back.num_features*2
        # head
        self.head = nn.Sequential(
            nn.Linear(g_dim + l_dim + 2, 1024),
            nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, ncls))

    def forward(self, g_img, l_imgs, depth):
        g_feat = self.g_back(g_img)                  # (B, g_dim)
        B,T = l_imgs.shape[:2]
        l_imgs = l_imgs.flatten(0,1)                 # (B*T,3,224,224)
        l_feat = self.l_back(l_imgs).view(B,T,-1)    # (B,T,l_dim/2)
        l_feat = torch.cat([l_feat.mean(1), l_feat.max(1).values], 1)
        z = torch.cat([g_feat, l_feat, depth], 1)
        return self.head(z)

# ───────────── MoCo-v2 utils (very light) ────────────────────────────────
@torch.no_grad()
def _momentum_update(q_encoder, k_encoder, m=0.999):
    for q,p in zip(q_encoder.parameters(), k_encoder.parameters()):
        p.data = p.data*m + q.data*(1.-m)

class MemoryQueue:
    def __init__(self, dim=256, K=65536):
        self.K=K; self.register = torch.randn(K,dim)
        self.ptr = 0
    def enqueue(self, x:torch.Tensor):
        n = x.size(0)
        if n>self.K: x=x[:self.K]
        if self.ptr+n<=self.K:
            self.register[self.ptr:self.ptr+n] = x.detach().cpu()
            self.ptr = (self.ptr+n)%self.K
        else:
            first = self.K - self.ptr
            self.register[self.ptr:]   = x[:first].detach().cpu()
            self.register[:n-first]    = x[first:].detach().cpu()
            self.ptr = n-first
    def get(self): return self.register.to(DEV)

# ───────────── pre-train (Swin only) ─────────────────────────────────────
def moco_pretrain(paths):
    q_enc = timm.create_model('swinv2_base_window12to16_224to384_22kft1k',
                              pretrained=True).to(DEV)
    k_enc = timm.create_model('swinv2_base_window12to16_224to384_22kft1k',
                              pretrained=True).to(DEV).eval()
    for p in k_enc.parameters(): p.requires_grad=False

    proj_q = nn.Sequential(nn.Linear(q_enc.num_features,256, bias=False),
                           nn.BatchNorm1d(256)).to(DEV)
    proj_k = nn.Sequential(nn.Linear(k_enc.num_features,256, bias=False),
                           nn.BatchNorm1d(256)).to(DEV)
    mem = MemoryQueue()

    optim_q = optim.SGD(list(q_enc.parameters())+list(proj_q.parameters()),
                        lr=0.03, momentum=0.9, weight_decay=1e-4)
    scaler=GradScaler()

    dl = DataLoader(SSLDataset(paths), batch_size=args.batch_ssl,
                    shuffle=True, num_workers=8, pin_memory=True,
                    drop_last=True)
    ce = nn.CrossEntropyLoss()

    for ep in range(args.epochs_ssl):
        pbar = tqdm(dl, desc=f'MoCo {ep+1}/{args.epochs_ssl}')
        for x1,x2 in pbar:
            x1,x2 = x1.to(DEV), x2.to(DEV)
            with autocast(AMP):
                q  = proj_q(q_enc(x1))
                k  = proj_k(k_enc(x2))
                q  = nn.functional.normalize(q, dim=1)
                k  = nn.functional.normalize(k, dim=1)

                logits_pos = (q * k.detach()).sum(1, keepdim=True)      # B×1
                logits_neg = q @ mem.get().T                            # B×K
                logits = torch.cat([logits_pos, logits_neg], 1) / 0.2
                labels = torch.zeros(x1.size(0), dtype=torch.long, device=DEV)
                loss   = ce(logits, labels)

            scaler.scale(loss).backward()
            if (pbar.n+1) % args.accum == 0:
                scaler.step(optim_q); scaler.update(); optim_q.zero_grad()

            _momentum_update(q_enc, k_enc)
            _momentum_update(proj_q, proj_k)
            mem.enqueue(k.detach())

    torch.save(q_enc.state_dict(), 'checkpoints/moco_swin.pt')
    return q_enc.num_features

# ───────────── supervised helper ─────────────────────────────────────────
def one_cycle(opt, epochs, steps):
    return optim.lr_scheduler.OneCycleLR(opt, 3e-4,
                                         epochs=epochs,
                                         steps_per_epoch=steps)

# ───────────── train / eval fold ─────────────────────────────────────────
def finetune_fold(train_idx, val_idx, fold_id):
    # datasets
    tr_i, tr_l = [IMGS[i] for i in train_idx], [LABELS[i] for i in train_idx]
    va_i, va_l = [IMGS[i] for i in val_idx],  [LABELS[i] for i in val_idx]
    tr_ds = SupDataset(tr_i, tr_l, train=True)
    va_ds = SupDataset(va_i, va_l, train=False)

    model = Net(NCLS).to(DEV)
    if os.path.isfile('checkpoints/moco_swin.pt'):
        sd = torch.load('checkpoints/moco_swin.pt', map_location=DEV)
        miss = model.g_back.load_state_dict(sd, strict=False)
        print('MoCo weights loaded; missing:', miss.missing_keys[:5])

    # loss / opt
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = one_cycle(opt, args.epochs_sup,
                      math.ceil(len(tr_ds)/args.batch_sup/args.accum))
    scaler = GradScaler()
    f1 = MulticlassF1Score(NCLS, average='macro').to(DEV)
    ema = Net(NCLS).to(DEV).eval()
    for p in ema.parameters(): p.requires_grad=False

    best_f1, best_sd = 0., None
    for ep in range(args.epochs_sup):
        # ---------- train ----------
        model.train(); loader = DataLoader(tr_ds, batch_size=args.batch_sup,
                                           shuffle=True, num_workers=8,
                                           pin_memory=True, drop_last=True)
        for i,(g,limgs,y,d) in enumerate(tqdm(loader, desc=f'F{fold_id} ep{ep+1}')):
            g,limgs,y,d = g.to(DEV), limgs.to(DEV), y.to(DEV), d.to(DEV)
            with autocast(AMP):
                loss = nn.functional.cross_entropy(model(g,limgs,d), y,
                                                   label_smoothing=0.05)
            scaler.scale(loss/args.accum).backward()
            if (i+1)%args.accum==0:
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
                # EMA
                with torch.no_grad():
                    for w_ema, w in zip(ema.parameters(), model.parameters()):
                        w_ema.data.mul_(0.995).add_(w.data, alpha=0.005)

        # ---------- val ------------
        model.eval(); vp,gt = [],[]
        val_loader = DataLoader(va_ds, batch_size=args.batch_sup,
                                shuffle=False, num_workers=4, pin_memory=True)
        with torch.no_grad(), autocast(AMP):
            for g,limgs,y,d in val_loader:
                vp.append(ema(g.to(DEV), limgs.to(DEV), d.to(DEV)).softmax(1))
                gt.append(y.to(DEV))
        vf1 = f1(torch.cat(vp), torch.cat(gt)).item()
        print(f'  val F1 = {vf1:.3f}')
        if vf1>best_f1: best_f1,best_sd=vf1,ema.state_dict().copy()
    torch.save(best_sd, f'checkpoints/fold{fold_id}.pt')
    return best_sd

# ───────────── orchestrate folds ─────────────────────────────────────────
skf = StratifiedKFold(args.folds, shuffle=True, random_state=SEED)
ALL_P, ALL_T = [], []
for k,(tr,va) in enumerate(skf.split(IMGS,LABELS),1):
    print(f'\n== Fold {k}/{args.folds} ==')
    if args.stage in ('ssl', 'all') and not os.path.isfile('checkpoints/moco_swin.pt'):
        moco_pretrain([IMGS[i] for i in tr])          # one-off; first fold
    sd = finetune_fold(tr, va, k)
    # evaluation
    model = Net(NCLS).to(DEV); model.load_state_dict(sd); model.eval()
    loader = DataLoader(SupDataset([IMGS[i] for i in va],
                                   [LABELS[i] for i in va], train=False),
                        batch_size=args.batch_sup)
    with torch.no_grad(), autocast(AMP):
        for g,limgs,y,d in loader:
            ALL_P.append(model(g.to(DEV), limgs.to(DEV), d.to(DEV)).softmax(1))
            ALL_T.append(y.to(DEV))

# ───────────── report ────────────────────────────────────────────────────
PRED = torch.cat(ALL_P).argmax(1).cpu().numpy()
TRUE = torch.cat(ALL_T).cpu().numpy()
print('\n===== 5-fold ensemble (EMA checkpoints) =====')
print(classification_report(TRUE, PRED, digits=4))
print(confusion_matrix(TRUE, PRED))
