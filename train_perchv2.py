#!/usr/bin/env python3
"""
train_perchv2.py
Standalone Python script converted from birdclef-perchv2-training.ipynb.
Refactored to run locally with automatically downloaded Perch models/assets.
"""

import os
import re
import gc
import time
import json
import random
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import kagglehub
import onnxruntime as ort
import matplotlib.pyplot as plt

# Disable warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────────────────────────────────────
# PATHS AND CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# Local project root path
BASE = Path("/Users/ushnesha/NextDrive/Documents/Academic/ML Learning/BirdClef")
SAVE_DIR = BASE / "birdclef2026_trained"
WORK_DIR = BASE / "cache"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

print("Checking/Downloading Perch model scientific names/labels via kagglehub...")
# Downloads the Perch TF2 model directory to get the assets/labels.csv
MODEL_DIR = Path(kagglehub.model_download("google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"))
print(f"Perch model assets directory: {MODEL_DIR}")

print("Checking/Downloading Perch ONNX model via kagglehub...")
# Downloads the Perch ONNX files dataset
ONNX_DIR = Path(kagglehub.dataset_download("rishikeshjani/perch-onnx-for-birdclef-2026"))
print(f"Perch ONNX model directory: {ONNX_DIR}")

# Locate ONNX file
ONNX_PERCH_PATH = next(ONNX_DIR.glob("**/perch_v2_no_dft*.onnx"),
                       next(ONNX_DIR.glob("**/perch_v2*.onnx"), Path("")))
assert ONNX_PERCH_PATH.exists(), f"Could not find Perch ONNX file in {ONNX_DIR}"
print(f"Found Perch ONNX model: {ONNX_PERCH_PATH}")

# Audio Constants
SR             = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES   = 60 * SR
N_WINDOWS      = 12

# Training Hyperparameters
PROTO_EPOCHS, PROTO_PATIENCE = 80, 20
RES_EPOCHS,   RES_PATIENCE   = 40, 12
PROTO_LR, RES_LR             = 1e-3, 1e-3
ENSEMBLE_W       = 0.5      # ProtoSSM vs (prior+MLP) blend
ALPHA_BLEND      = 0.4      # MLP-probe blend
CORRECTION_WEIGHT= 0.30     # ResidualSSM correction
N_SITES_CAP      = 20

# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION & SETUP
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Torch device: {DEVICE}")

# Initialize ONNX Runtime Session
_prov = ort.get_available_providers()
providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
             if "CUDAExecutionProvider" in _prov else ["CPUExecutionProvider"])
_so = ort.SessionOptions()
_so.intra_op_num_threads = 4
ONNX_SESSION = ort.InferenceSession(str(ONNX_PERCH_PATH), sess_options=_so, providers=providers)
ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
ONNX_OUT_MAP = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
print("Perch via ONNX on:", ONNX_SESSION.get_providers())

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("Loading datasets...")
taxonomy          = pd.read_csv(BASE / "taxonomy.csv")
sample_sub        = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\\.ogg")
def parse_fname(name):
    m = FNAME_RE.match(name)
    if not m: return {"site": "unknown", "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}

def union_labels(series):
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t: out.add(t)
    return sorted(out)

# Align labels and soundscape data
sc = (soundscape_labels
      .groupby(["filename", "start", "end"])["primary_label"]
      .apply(union_labels).reset_index(name="label_list"))
sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
sc["row_id"]  = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)
_meta = sc["filename"].apply(parse_fname).apply(pd.Series)
sc = pd.concat([sc, _meta], axis=1)

Y_SC = np.zeros((len(sc), N_CLASSES), dtype=np.uint8)
for i, lbls in enumerate(sc["label_list"]):
    for lbl in lbls:
        if lbl in label_to_idx: Y_SC[i, label_to_idx[lbl]] = 1

windows_per_file = sc.groupby("filename").size()
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc["fully_labeled"] = sc["filename"].isin(full_files)
full_rows = (sc[sc["fully_labeled"]].sort_values(["filename", "end_sec"]).reset_index(drop=False))
Y_FULL = Y_SC[full_rows["index"].to_numpy()]
print(f"Classes: {N_CLASSES} | Fully-labeled files: {len(full_files)} | windows: {len(full_rows)}")

# Perch output alignment using assets/labels.csv
bc_labels = (pd.read_csv(MODEL_DIR / "assets" / "labels.csv").reset_index()
             .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}))
NO_LABEL = len(bc_labels)
mapping = taxonomy.merge(bc_labels, on="scientific_name", how="left")
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
lbl2bc = mapping.set_index("primary_label")["bc_index"]

BC_INDICES    = np.array([int(lbl2bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK   = BC_INDICES != NO_LABEL
MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)
CLASS_NAME_MAP= taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA  = {"Amphibia", "Insecta"}
print(f"Mapped: {MAPPED_MASK.sum()} / {N_CLASSES} species have a Perch logit")

# Mapping unmapped classes to proxy genus logits
proxy_map = {}
unmapped_df = taxonomy[taxonomy["primary_label"].isin([PRIMARY_LABELS[i] for i in UNMAPPED_POS])].copy()
for _, row in unmapped_df.iterrows():
    genus = str(row["scientific_name"]).split()[0]
    hits = bc_labels[bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\\s", na=False)]
    if len(hits) > 0:
        proxy_map[label_to_idx[row["primary_label"]]] = hits["bc_index"].astype(int).tolist()
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
proxy_map  = {idx: v for idx, v in proxy_map.items()
              if CLASS_NAME_MAP.get(PRIMARY_LABELS[idx]) in PROXY_TAXA}
print(f"Proxy classes: {len(proxy_map)}")

# Dynamic temperature scaling
temperatures = np.ones(N_CLASSES, dtype=np.float32)
for ci, label in enumerate(PRIMARY_LABELS):
    temperatures[ci] = 0.95 if CLASS_NAME_MAP.get(label, "Aves") in TEXTURE_TAXA else 1.10

# ─────────────────────────────────────────────────────────────────────────────
# CACHING AND RUNNING PERCH ON ONNX
# ─────────────────────────────────────────────────────────────────────────────
import concurrent.futures
def read_60s(path):
    y, _ = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES: y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:                     y = y[:FILE_SAMPLES]
    return y

def run_perch(paths, batch_files=16, verbose=True):
    paths = [Path(p) for p in paths]; n_rows = len(paths) * N_WINDOWS
    row_ids=np.empty(n_rows,object); filenames=np.empty(n_rows,object)
    sites=np.empty(n_rows,object);   hours=np.zeros(n_rows,np.int16)
    scores=np.zeros((n_rows,N_CLASSES),np.float32); embs=np.zeros((n_rows,1536),np.float32)
    wr=0
    itr=tqdm(range(0,len(paths),batch_files),desc="Perch") if verbose else range(0,len(paths),batch_files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io:
        next_paths=paths[0:batch_files]; fut=[io.submit(read_60s,p) for p in next_paths]
        for start in itr:
            bp=next_paths; bn=len(bp); ba=[f.result() for f in fut]
            ns=start+batch_files
            if ns<len(paths):
                next_paths=paths[ns:ns+batch_files]; fut=[io.submit(read_60s,p) for p in next_paths]
            x=np.empty((bn*N_WINDOWS,WINDOW_SAMPLES),np.float32); br=wr
            for bi,path in enumerate(bp):
                y=ba[bi]; meta=parse_fname(path.name); stem=path.stem
                x[bi*N_WINDOWS:(bi+1)*N_WINDOWS]=y.reshape(N_WINDOWS,WINDOW_SAMPLES)
                row_ids[wr:wr+N_WINDOWS]=[f"{stem}_{t}" for t in range(5,65,5)]
                filenames[wr:wr+N_WINDOWS]=path.name; sites[wr:wr+N_WINDOWS]=meta["site"]
                hours[wr:wr+N_WINDOWS]=meta["hour_utc"]; wr+=N_WINDOWS
            outs   = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
            logits = outs[ONNX_OUT_MAP["label"]].astype(np.float32)
            emb    = outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)
            scores[br:wr,MAPPED_POS]=logits[:,MAPPED_BC_IDX]; embs[br:wr]=emb
            for pos_idx,bc_idxs in proxy_map.items():
                scores[br:wr,pos_idx]=logits[:,np.array(bc_idxs,np.int32)].max(axis=1)
            del x,logits,emb,ba; gc.collect()
    meta_df=pd.DataFrame({"row_id":row_ids,"filename":filenames,"site":sites,"hour_utc":hours})
    return meta_df, scores, embs

print("\nBuilding caching feature vectors on local train soundscapes...")
train_paths=[BASE/"train_soundscapes"/fn for fn in full_files]
train_paths=[p for p in train_paths if p.exists()]
assert len(train_paths) > 0, "No soundscape audio files found in train_soundscapes! Check dataset path."

t0=time.time()
meta_tr, sc_tr, emb_tr = run_perch(train_paths, batch_files=4, verbose=True)
print(f"Perch cache: {time.time()-t0:.1f}s  scores={sc_tr.shape}  embs={emb_tr.shape}")

row_id_to_index = full_rows.set_index("row_id")["index"]
Y_FULL_aligned  = Y_SC[row_id_to_index.loc[meta_tr["row_id"]].to_numpy()]
print("Y_FULL_aligned:", Y_FULL_aligned.shape)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS & ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-30,30)))

# MLP probe helpers
def build_class_freq_weights(Y, cap=10.0):
    pos=Y.sum(0).astype(np.float32)+1.0; freq=pos/Y.shape[0]
    w=np.clip(1.0/(freq**0.5),1.0,cap); return (w/w.mean()).astype(np.float32)

def build_sequential_features(scores_col, n_windows=12):
    x=scores_col.reshape(-1,n_windows)
    prev=np.concatenate([x[:,:1],x[:,:-1]],1); next_=np.concatenate([x[:,1:],x[:,-1:]],1)
    mean=np.repeat(x.mean(1),n_windows); max_=np.repeat(x.max(1),n_windows); std=np.repeat(x.std(1),n_windows)
    return prev.reshape(-1),next_.reshape(-1),mean,max_,std

def train_mlp_probes(emb, scores_raw, Y, min_pos=5, pca_dim=64, alpha_blend=0.4):
    scaler=StandardScaler(); emb_s=scaler.fit_transform(emb)
    pca=PCA(n_components=min(pca_dim,emb_s.shape[1]-1)); Z=pca.fit_transform(emb_s).astype(np.float32)
    print(f"PCA: {Z.shape} var={pca.explained_variance_ratio_.sum():.2%}")
    cw=build_class_freq_weights(Y,cap=10.0); probe={}; active=np.where(Y.sum(0)>=min_pos)[0]; MAX_ROWS=3000
    for ci in tqdm(active,desc="MLP probes"):
        y=Y[:,ci]
        if y.sum()==0 or y.sum()==len(y): continue
        prev,next_,mean,max_,std=build_sequential_features(scores_raw[:,ci])
        X=np.hstack([Z,scores_raw[:,ci:ci+1],prev[:,None],next_[:,None],mean[:,None],max_[:,None],std[:,None]])
        n_pos=int(y.sum()); n_neg=len(y)-n_pos; pos_idx=np.where(y==1)[0]
        w=float(cw[ci]); rep=max(1,min(int(round(w*n_neg/max(n_pos,1))),8))
        if n_pos*rep+len(y)>MAX_ROWS: rep=max(1,(MAX_ROWS-len(y))//max(n_pos,1))
        Xb=np.vstack([X,np.tile(X[pos_idx],(rep,1))]); yb=np.concatenate([y,np.ones(n_pos*rep,dtype=y.dtype)])
        clf=MLPClassifier(hidden_layer_sizes=(128,64),activation="relu",max_iter=300,early_stopping=True,
                          validation_fraction=0.15,n_iter_no_change=15,random_state=42,
                          learning_rate_init=5e-4,alpha=0.005)
        clf.fit(Xb,yb); probe[ci]=clf
    print(f"Trained {len(probe)} MLP probes")
    return probe, scaler, pca, alpha_blend

class VectorizedMLPProbes(nn.Module):
    def __init__(self, probe_models):
        super().__init__(); self.valid_classes=sorted(probe_models.keys()); V=len(self.valid_classes)
        if V==0: self.weights=nn.ParameterList(); self.biases=nn.ParameterList(); self.n_layers=0; return
        s=probe_models[self.valid_classes[0]]; self.n_layers=len(s.coefs_)
        self.weights=nn.ParameterList(); self.biases=nn.ParameterList()
        for li in range(self.n_layers):
            W=np.stack([probe_models[c].coefs_[li] for c in self.valid_classes],0)
            b=np.stack([probe_models[c].intercepts_[li] for c in self.valid_classes],0)
            self.weights.append(nn.Parameter(torch.tensor(W,dtype=torch.float32),requires_grad=False))
            self.biases.append(nn.Parameter(torch.tensor(b,dtype=torch.float32),requires_grad=False))
    def forward(self,x):
        h=x
        for i in range(self.n_layers):
            h=torch.bmm(h,self.weights[i])+self.biases[i].unsqueeze(1)
            if i<self.n_layers-1: h=torch.relu(h)
        return h.squeeze(-1)

def apply_mlp_probes_vectorized(emb_test, scores_test, probe_models, scaler, pca, alpha_blend=0.4):
    if len(probe_models)==0: return scores_test.copy()
    Z=pca.transform(scaler.transform(emb_test)).astype(np.float32); vc=sorted(probe_models.keys())
    V=len(vc); N=len(scores_test); raw=scores_test[:,vc].T; nf=N//N_WINDOWS
    rv=raw.reshape(V,nf,N_WINDOWS)
    prev=np.concatenate([rv[:,:,:1],rv[:,:,:-1]],2).reshape(V,N); nxt=np.concatenate([rv[:,:,1:],rv[:,:,-1:]],2).reshape(V,N)
    mean=np.repeat(rv.mean(2),N_WINDOWS,1); mx=np.repeat(rv.max(2),N_WINDOWS,1); std=np.repeat(rv.std(2),N_WINDOWS,1)
    sf_=np.stack([raw,prev,nxt,mean,mx,std],-1).astype(np.float32)
    Ze=np.broadcast_to(Z,(V,N,Z.shape[1])); X=np.concatenate([Ze.astype(np.float32),sf_],-1)
    vp=VectorizedMLPProbes(probe_models).eval()
    with torch.no_grad(): preds=vp(torch.tensor(X)).numpy()
    out=scores_test.copy(); out[:,vc]=(1.0-alpha_blend)*scores_test[:,vc]+alpha_blend*preds.T
    return out

# priors / calibration
def build_prior_tables(sc_df, Y_labels):
    sc_df=sc_df.reset_index(drop=True); gp=Y_labels.mean(0).astype(np.float32)
    sk=sorted(sc_df["site"].dropna().astype(str).unique()); s2i={k:i for i,k in enumerate(sk)}
    sp=np.zeros((len(sk),Y_labels.shape[1]),np.float32); sn=np.zeros(len(sk),np.float32)
    for s in sk:
        i=s2i[s]; m=sc_df["site"].astype(str).values==s; sn[i]=m.sum(); sp[i]=Y_labels[m].mean(0)
    hk=sorted(sc_df["hour_utc"].dropna().astype(int).unique()); h2i={h:i for i,h in enumerate(hk)}
    hp=np.zeros((len(hk),Y_labels.shape[1]),np.float32); hn=np.zeros(len(hk),np.float32)
    for h in hk:
        i=h2i[h]; m=sc_df["hour_utc"].astype(int).values==h; hn[i]=m.sum(); hp[i]=Y_labels[m].mean(0)
    shk=sorted({(str(s),int(h)) for s,h in zip(sc_df["site"].dropna(),sc_df["hour_utc"].dropna())})
    sh2i={k:i for i,k in enumerate(shk)}; shp=np.zeros((len(shk),Y_labels.shape[1]),np.float32); shn=np.zeros(len(shk),np.float32)
    for (s,h) in shk:
        i=sh2i[(s,h)]; m=(sc_df["site"].astype(str).values==s)&(sc_df["hour_utc"].astype(int).values==h); shn[i]=m.sum(); shp[i]=Y_labels[m].mean(0)
    return {"global_p":gp,"site_to_i":s2i,"site_p":sp,"site_n":sn,"hour_to_i":h2i,"hour_p":hp,"hour_n":hn,
            "sh_to_i":sh2i,"sh_p":shp,"sh_n":shn}

def apply_prior(scores, sites, hours, tables, lambda_prior=0.4):
    eps=1e-4; n=len(scores); out=scores.copy(); p=np.tile(tables["global_p"],(n,1))
    for i,h in enumerate(hours):
        h=int(h)
        if h in tables["hour_to_i"]:
            j=tables["hour_to_i"][h]; nh=tables["hour_n"][j]; w=nh/(nh+8.0); p[i]=w*tables["hour_p"][j]+(1-w)*tables["global_p"]
    for i,s in enumerate(sites):
        s=str(s)
        if s in tables["site_to_i"]:
            j=tables["site_to_i"][s]; ns=tables["site_n"][j]; w=ns/(ns+8.0); p[i]=w*tables["site_p"][j]+(1-w)*p[i]
    for i,(s,h) in enumerate(zip(sites,hours)):
        key=(str(s),int(h))
        if key in tables["sh_to_i"]:
            j=tables["sh_to_i"][key]; nsh=tables["sh_n"][j]; w=nsh/(nsh+4.0); p[i]=w*tables["sh_p"][j]+(1-w)*p[i]
    p=np.clip(p,eps,1-eps); out+=lambda_prior*(np.log(p)-np.log1p(-p)); return out.astype(np.float32)

def calibrate_and_optimize_thresholds(oof_probs, Y_FULL, threshold_grid=None, n_windows=12):
    if threshold_grid is None: threshold_grid=[0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]
    n,nc=oof_probs.shape; thr=np.full(nc,0.5,np.float32); nf=n//n_windows
    fo=oof_probs.reshape(nf,n_windows,nc).max(1); fy=Y_FULL.reshape(nf,n_windows,nc).max(1); nca=0
    for c in range(nc):
        yt=fy[:,c]; yp=fo[:,c]
        if yt.sum()<3: continue
        try: ir=IsotonicRegression(out_of_bounds="clip"); ir.fit(yp,yt); yc=ir.transform(yp)
        except: yc=yp
        bf,bt=0.0,0.5
        for t in threshold_grid:
            pr=(yc>=t).astype(int); tp=((pr==1)&(yt==1)).sum(); fp=((pr==1)&(yt==0)).sum(); fn=((pr==0)&(yt==1)).sum()
            pre=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8); f1=2*pre*rec/(pre+rec+1e-8)
            if f1>bf: bf,bt=f1,t
        thr[c]=bt; nca+=1
    print(f"Calibrated {nca} classes | mean thr {thr.mean():.3f}")
    return thr

# SSM architecture
class SelectiveSSM(nn.Module):
    def __init__(self,d_model,d_state=16,d_conv=4):
        super().__init__(); self.d_model=d_model; self.d_state=d_state
        self.in_proj=nn.Linear(d_model,2*d_model,bias=False)
        self.conv1d=nn.Conv1d(d_model,d_model,d_conv,padding=d_conv-1,groups=d_model)
        self.dt_proj=nn.Linear(d_model,d_model,bias=True)
        A=torch.arange(1,d_state+1,dtype=torch.float32).unsqueeze(0).expand(d_model,-1)
        self.A_log=nn.Parameter(torch.log(A)); self.D=nn.Parameter(torch.ones(d_model))
        self.B_proj=nn.Linear(d_model,d_state,bias=False); self.C_proj=nn.Linear(d_model,d_state,bias=False)
        self.out_proj=nn.Linear(d_model,d_model,bias=False)
    def forward(self,x):
        B_sz,T,D=x.shape; xz=self.in_proj(x); x_ssm,z=xz.chunk(2,-1)
        x_conv=F.silu(self.conv1d(x_ssm.transpose(1,2))[:,:,:T].transpose(1,2))
        dt=F.softplus(self.dt_proj(x_conv)); A=-torch.exp(self.A_log)
        B=self.B_proj(x_conv); C=self.C_proj(x_conv)
        h=torch.zeros(B_sz,D,self.d_state,device=x.device); ys=[]
        for t in range(T):
            dA=torch.exp(A[None]*dt[:,t,:,None]); dB=dt[:,t,:,None]*B[:,t,None,:]
            h=h*dA+x[:,t,:,None]*dB; ys.append((h*C[:,t,None,:]).sum(-1))
        return torch.stack(ys,1)+x*self.D[None,None,:]

class LightProtoSSM(nn.Module):
    def __init__(self,d_input=1536,d_model=128,d_state=16,n_classes=234,n_windows=12,
                 dropout=0.15,n_sites=20,meta_dim=16,use_cross_attn=True,cross_attn_heads=2):
        super().__init__(); self.n_classes=n_classes; self.n_windows=n_windows; self.use_cross_attn=use_cross_attn
        self.input_proj=nn.Sequential(nn.Linear(d_input,d_model),nn.LayerNorm(d_model),nn.GELU(),nn.Dropout(dropout))
        self.pos_enc=nn.Parameter(torch.randn(1,n_windows,d_model)*0.02)
        self.site_emb=nn.Embedding(n_sites,meta_dim); self.hour_emb=nn.Embedding(24,meta_dim)
        self.meta_proj=nn.Linear(2*meta_dim,d_model)
        self.ssm_fwd=nn.ModuleList([SelectiveSSM(d_model,d_state) for _ in range(2)])
        self.ssm_bwd=nn.ModuleList([SelectiveSSM(d_model,d_state) for _ in range(2)])
        self.ssm_merge=nn.ModuleList([nn.Linear(2*d_model,d_model) for _ in range(2)])
        self.ssm_norm=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)]); self.drop=nn.Dropout(dropout)
        if use_cross_attn:
            self.cross_attn=nn.ModuleList([nn.MultiheadAttention(d_model,cross_attn_heads,dropout=dropout,batch_first=True) for _ in range(2)])
            self.cross_norm=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.prototypes=nn.Parameter(torch.randn(n_classes,d_model)*0.02)
        self.proto_temp=nn.Parameter(torch.tensor(5.0)); self.class_bias=nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha=nn.Parameter(torch.zeros(n_classes))
    def init_prototypes(self,emb_tensor,labels_tensor):
        with torch.no_grad():
            h=self.input_proj(emb_tensor)
            for c in range(self.n_classes):
                m=labels_tensor[:,c]>0.5
                if m.sum()>0: self.prototypes.data[c]=F.normalize(h[m].mean(0),dim=0)
    def forward(self,emb,perch_logits=None,site_ids=None,hours=None):
        B,T,_=emb.shape; h=self.input_proj(emb)+self.pos_enc[:,:T,:]
        if site_ids is not None and hours is not None:
            meta=self.meta_proj(torch.cat([self.site_emb(site_ids),self.hour_emb(hours)],-1)); h=h+meta[:,None,:]
        for i,(fwd,bwd,merge,norm) in enumerate(zip(self.ssm_fwd,self.ssm_bwd,self.ssm_merge,self.ssm_norm)):
            res=h; hf=fwd(h); hb=bwd(h.flip(1)).flip(1); h=self.drop(merge(torch.cat([hf,hb],-1))); h=norm(h+res)
            if self.use_cross_attn:
                a,_=self.cross_attn[i](h,h,h); h=self.cross_norm[i](h+a)
        hn=F.normalize(h,dim=-1); pn=F.normalize(self.prototypes,dim=-1)
        sim=torch.matmul(hn,pn.T)*F.softplus(self.proto_temp)+self.class_bias[None,None,:]
        if perch_logits is not None:
            alpha=torch.sigmoid(self.fusion_alpha)[None,None,:]; out=alpha*sim+(1-alpha)*perch_logits
        else: out=sim
        return out

class ResidualSSM(nn.Module):
    def __init__(self,d_input=1536,d_scores=234,d_model=64,d_state=8,n_classes=234,
                 n_windows=12,dropout=0.1,n_sites=20,meta_dim=8):
        super().__init__(); self.n_classes=n_classes
        self.input_proj=nn.Sequential(nn.Linear(d_input+d_scores,d_model),nn.LayerNorm(d_model),nn.GELU(),nn.Dropout(dropout))
        self.site_emb=nn.Embedding(n_sites,meta_dim); self.hour_emb=nn.Embedding(24,meta_dim)
        self.meta_proj=nn.Linear(2*meta_dim,d_model); self.pos_enc=nn.Parameter(torch.randn(1,n_windows,d_model)*0.02)
        self.ssm_fwd=SelectiveSSM(d_model,d_state); self.ssm_bwd=SelectiveSSM(d_model,d_state)
        self.ssm_merge=nn.Linear(2*d_model,d_model); self.ssm_norm=nn.LayerNorm(d_model); self.ssm_drop=nn.Dropout(dropout)
        self.output_head=nn.Linear(d_model,n_classes); nn.init.zeros_(self.output_head.weight); nn.init.zeros_(self.output_head.bias)
    def forward(self,emb,first_pass,site_ids=None,hours=None):
        B,T,_=emb.shape; x=torch.cat([emb,first_pass],-1); h=self.input_proj(x)+self.pos_enc[:,:T,:]
        if site_ids is not None and hours is not None:
            meta=self.meta_proj(torch.cat([self.site_emb(site_ids.clamp(0,self.site_emb.num_embeddings-1)),
                                           self.hour_emb(hours.clamp(0,23))],-1)); h=h+meta.unsqueeze(1)
        res=h; hf=self.ssm_fwd(h); hb=self.ssm_bwd(h.flip(1)).flip(1)
        h=self.ssm_drop(self.ssm_merge(torch.cat([hf,hb],-1))); h=self.ssm_norm(h+res)
        return self.output_head(h)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOPS
# ─────────────────────────────────────────────────────────────────────────────
def train_light_proto_ssm(emb_full,scores_full,Y_full,meta_full,n_epochs,patience,lr,n_sites=20,device="cpu"):
    nf=len(emb_full)//N_WINDOWS; emb_f=emb_full.reshape(nf,N_WINDOWS,-1)
    log_f=scores_full.reshape(nf,N_WINDOWS,-1); lab_f=Y_full.reshape(nf,N_WINDOWS,-1).astype(np.float32)
    fnames=meta_full["filename"].unique(); sites_u=sorted(meta_full["site"].unique()); s2i={s:i+1 for i,s in enumerate(sites_u)}
    site_ids=np.array([min(s2i.get(meta_full.loc[meta_full["filename"]==fn,"site"].iloc[0],0),n_sites-1) for fn in fnames],np.int64)
    hour_ids=np.array([int(meta_full.loc[meta_full["filename"]==fn,"hour_utc"].iloc[0])%24 for fn in fnames],np.int64)
    
    n_val=max(1,int(nf*0.15)); rng=torch.Generator(); rng.manual_seed(42)
    perm=torch.randperm(nf,generator=rng).numpy(); vi=perm[:n_val]; ti=perm[n_val:]
    
    model=LightProtoSSM(n_classes=N_CLASSES,n_sites=n_sites,use_cross_attn=True,cross_attn_heads=2)
    model.init_prototypes(torch.tensor(emb_full,dtype=torch.float32),torch.tensor(Y_full,dtype=torch.float32))
    model=model.to(device)
    emb_t=torch.tensor(emb_f,dtype=torch.float32,device=device); log_t=torch.tensor(log_f,dtype=torch.float32,device=device)
    lab_t=torch.tensor(lab_f,dtype=torch.float32,device=device)
    site_t=torch.tensor(site_ids,dtype=torch.long,device=device); hour_t=torch.tensor(hour_ids,dtype=torch.long,device=device)
    pc=lab_t.sum(dim=(0,1)); tot=lab_t.shape[0]*lab_t.shape[1]; pw=((tot-pc)/(pc+1)).clamp(max=25.0)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,epochs=n_epochs,steps_per_epoch=1,pct_start=0.1,anneal_strategy="cos")
    best,bs,wait=float("inf"),None,0
    swa=torch.optim.swa_utils.AveragedModel(model); swa_start=int(n_epochs*0.65)
    swa_sched=torch.optim.swa_utils.SWALR(opt,swa_lr=4e-4)
    
    train_history = []
    val_history = []
    
    for ep in range(n_epochs):
        model.train()
        out=model(emb_t[ti],log_t[ti],site_ids=site_t[ti],hours=hour_t[ti])
        loss=F.binary_cross_entropy_with_logits(out,lab_t[ti],pos_weight=pw[None,None,:])+0.15*F.mse_loss(out,log_t[ti])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if ep>=swa_start: swa.update_parameters(model); swa_sched.step()
        else: sched.step()
        
        # Validation loss tracking
        model.eval()
        with torch.no_grad():
            val_out=model(emb_t[vi],log_t[vi],site_ids=site_t[vi],hours=hour_t[vi])
            val_loss=F.binary_cross_entropy_with_logits(val_out,lab_t[vi],pos_weight=pw[None,None,:])+0.15*F.mse_loss(val_out,log_t[vi])
        
        train_history.append(loss.item())
        val_history.append(val_loss.item())
        
        if val_loss.item()<best: best=val_loss.item(); bs={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=patience: break
            
    if ep>=swa_start:
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0),swa); model=swa
    else: model.load_state_dict(bs)
    model.eval(); return model,s2i,(train_history, val_history)

def run_tta_proto(model,emb_files,sc_files,site_t,hour_t,shifts=[0,1,-1,2,-2],device="cpu"):
    model.eval(); preds=[]
    emb_t=torch.tensor(emb_files,dtype=torch.float32,device=device); sc_t=torch.tensor(sc_files,dtype=torch.float32,device=device)
    st=site_t.to(device); ht=hour_t.to(device)
    for sh in shifts:
        e=torch.roll(emb_t,sh,dims=1) if sh else emb_t; s=torch.roll(sc_t,sh,dims=1) if sh else sc_t
        with torch.no_grad(): out=model(e,s,site_ids=st,hours=ht).cpu().numpy()
        if sh: out=np.roll(out,-sh,axis=1)
        preds.append(out)
    return np.mean(preds,0)

def train_residual_ssm(emb_full,first_pass_flat,Y_full,site_ids,hour_ids,n_epochs,patience,lr,correction_weight,device="cpu"):
    nf=len(emb_full)//N_WINDOWS; emb_f=emb_full.reshape(nf,N_WINDOWS,-1)
    fp_f=first_pass_flat.reshape(nf,N_WINDOWS,-1); lab_f=Y_full.reshape(nf,N_WINDOWS,-1).astype(np.float32)
    fp_prob=1.0/(1.0+np.exp(-np.clip(fp_f,-30,30))); residuals=lab_f-fp_prob
    n_val=max(1,int(nf*0.15)); rng=torch.Generator(); rng.manual_seed(42)
    perm=torch.randperm(nf,generator=rng).numpy(); vi=perm[:n_val]; ti=perm[n_val:]
    emb_t=torch.tensor(emb_f,dtype=torch.float32,device=device); fp_t=torch.tensor(fp_f,dtype=torch.float32,device=device)
    res_t=torch.tensor(residuals,dtype=torch.float32,device=device)
    site_t=torch.tensor(site_ids,dtype=torch.long,device=device); hour_t=torch.tensor(hour_ids,dtype=torch.long,device=device)
    model=ResidualSSM(n_classes=N_CLASSES).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,epochs=n_epochs,steps_per_epoch=1,pct_start=0.1,anneal_strategy="cos")
    best,bs,wait=float("inf"),None,0
    
    train_history = []
    val_history = []
    
    for ep in range(n_epochs):
        model.train(); corr=model(emb_t[ti],fp_t[ti],site_ids=site_t[ti],hours=hour_t[ti]); loss=F.mse_loss(corr,res_t[ti])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            vc=model(emb_t[vi],fp_t[vi],site_ids=site_t[vi],hours=hour_t[vi]); vl=F.mse_loss(vc,res_t[vi])
            
        train_history.append(loss.item())
        val_history.append(vl.item())
        
        if vl.item()<best: best=vl.item(); bs={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=patience: break
    model.load_state_dict(bs); return model,correction_weight,(train_history, val_history)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION: TRAIN ALL MODELS & SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1) ProtoSSM
    print("\n[Phase 1] Training ProtoSSM Model...")
    t0=time.time()
    proto_model, site2i_tr, proto_history = train_light_proto_ssm(
        emb_tr, sc_tr, Y_FULL_aligned, meta_tr,
        n_epochs=PROTO_EPOCHS, patience=PROTO_PATIENCE, lr=PROTO_LR, n_sites=N_SITES_CAP, device=DEVICE)
    print(f"ProtoSSM trained: {time.time()-t0:.1f}s")

    # 2) Priors + MLP probes
    print("\n[Phase 2] Training Priors and MLP Probes...")
    prior_tables = build_prior_tables(sc, Y_SC)
    probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
        emb_tr, sc_tr, Y_FULL_aligned, min_pos=5, pca_dim=64, alpha_blend=ALPHA_BLEND)

    # 3) Train-side first pass (for threshold calibration)
    print("\n[Phase 3] Generating first-pass predictions for calibration...")
    tr_fnames = meta_tr.drop_duplicates("filename")["filename"].tolist()
    tr_site_ids=np.array([min(site2i_tr.get(meta_tr.loc[meta_tr["filename"]==fn,"site"].iloc[0],0),N_SITES_CAP-1) for fn in tr_fnames],np.int64)
    tr_hour_ids=np.array([int(meta_tr.loc[meta_tr["filename"]==fn,"hour_utc"].iloc[0])%24 for fn in tr_fnames],np.int64)
    nf=len(sc_tr)//N_WINDOWS; emb_tr_f=emb_tr.reshape(nf,N_WINDOWS,-1); sc_tr_f=sc_tr.reshape(nf,N_WINDOWS,-1)
    
    proto_tr_flat = run_tta_proto(proto_model, emb_tr_f, sc_tr_f,
                                  torch.tensor(tr_site_ids,dtype=torch.long),
                                  torch.tensor(tr_hour_ids,dtype=torch.long),
                                  shifts=[0,1,-1,2,-2], device=DEVICE).reshape(-1,N_CLASSES).astype(np.float32)
    sc_tr_prior = apply_prior(sc_tr, meta_tr["site"].to_numpy(), meta_tr["hour_utc"].to_numpy(), prior_tables, 0.4)
    sc_tr_mlp   = apply_mlp_probes_vectorized(emb_tr, sc_tr_prior, probe_models, emb_scaler, emb_pca, alpha_blend)
    first_pass_tr = ENSEMBLE_W*proto_tr_flat + (1.0-ENSEMBLE_W)*sc_tr_mlp

    PER_CLASS_THRESHOLDS = calibrate_and_optimize_thresholds(sigmoid(first_pass_tr), Y_FULL_aligned, n_windows=N_WINDOWS)

    # 4) ResidualSSM
    print("\n[Phase 4] Training ResidualSSM Model...")
    t0=time.time()
    res_model, correction_weight, res_history = train_residual_ssm(
        emb_tr, first_pass_tr, Y_FULL_aligned, tr_site_ids, tr_hour_ids,
        n_epochs=RES_EPOCHS, patience=RES_PATIENCE, lr=RES_LR, correction_weight=CORRECTION_WEIGHT, device=DEVICE)
    print(f"ResidualSSM trained: {time.time()-t0:.1f}s")

    # ── PLOT PERFORMANCE CURVES ───────────────────────────────────────────────
    print("\nGenerating training vs validation performance curves...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. ProtoSSM Loss Curve
    plt.figure(figsize=(10, 4))
    plt.plot(proto_history[0], label="Train Loss", color="blue")
    plt.plot(proto_history[1], label="Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ProtoSSM Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    proto_plot_path = SAVE_DIR / "protossm_loss_curve.png"
    plt.savefig(proto_plot_path, dpi=300)
    plt.close()
    print(f"Saved ProtoSSM loss curve to: {proto_plot_path}")
    
    # 2. ResidualSSM Loss Curve
    plt.figure(figsize=(10, 4))
    plt.plot(res_history[0], label="Train MSE", color="blue")
    plt.plot(res_history[1], label="Val MSE", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("ResidualSSM Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    res_plot_path = SAVE_DIR / "residualssm_loss_curve.png"
    plt.savefig(res_plot_path, dpi=300)
    plt.close()
    print(f"Saved ResidualSSM loss curve to: {res_plot_path}")

    # ── SAVE ALL ARTIFACTS ────────────────────────────────────────────────────
    print("\nSaving all trained model checkpoints and configurations...")
    def proto_state(m): return (m.module if isinstance(m, AveragedModel) else m).state_dict()
    torch.save({k:v.cpu() for k,v in proto_state(proto_model).items()}, SAVE_DIR/"proto_ssm.pt")
    torch.save({k:v.cpu() for k,v in res_model.state_dict().items()},   SAVE_DIR/"residual_ssm.pt")
    joblib.dump(probe_models, SAVE_DIR/"mlp_probes.joblib")
    joblib.dump(emb_scaler,   SAVE_DIR/"emb_scaler.joblib")
    joblib.dump(emb_pca,      SAVE_DIR/"emb_pca.joblib")
    joblib.dump(prior_tables, SAVE_DIR/"prior_tables.joblib")
    np.save(SAVE_DIR/"thresholds.npy",   PER_CLASS_THRESHOLDS)
    np.save(SAVE_DIR/"temperatures.npy", temperatures)
    
    with open(SAVE_DIR/"meta.json", "w") as f:
        json.dump({"primary_labels":PRIMARY_LABELS,"n_classes":N_CLASSES,"n_windows":N_WINDOWS,
                   "n_sites":N_SITES_CAP,"ensemble_w":ENSEMBLE_W,"alpha_blend":alpha_blend,
                   "correction_weight":correction_weight,"site2i":site2i_tr,
                   "proto":{"d_model":128,"d_state":16,"meta_dim":16,"cross_attn_heads":2,"use_cross_attn":True},
                   "residual":{"d_model":64,"d_state":8,"meta_dim":8}}, f)
        
    print(f"Saved artifacts to {SAVE_DIR}")
    print("Files saved:")
    print(sorted(p.name for p in SAVE_DIR.iterdir()))
    print("\nTraining workflow completed successfully!")

if __name__ == "__main__":
    main()
