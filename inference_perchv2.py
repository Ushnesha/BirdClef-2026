#!/usr/bin/env python3
"""
inference_perchv2.py
Standalone Python script converted and updated from birdclef-inference-perchv2.ipynb.
Configured to use local directories instead of Kaggle paths, with automatic downloading
of external models using kagglehub if not present.
"""

import subprocess
import sys
import os
import re
import gc
import time
import warnings
import json
import importlib
import random
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import librosa
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm
import kagglehub
import onnxruntime as ort

# Disable warnings and tensorflow logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────────────────────────────────────
# PATHS AND CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
print(f"Local project root: {BASE}")

# Local trained artifacts directory
TRAINED = BASE / "birdclef2026_trained"
assert TRAINED.exists(), f"Could not find trained artifacts directory at {TRAINED}"

# Check/download Perch backbone assets (labels, etc.)
try:
    print("Checking/Downloading Perch model assets via kagglehub...")
    MODEL_DIR = Path(kagglehub.model_download("google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"))
    print(f"Perch model assets directory: {MODEL_DIR}")
except Exception as e:
    print(f"Warning: Failed to download Perch model via kagglehub: {e}")
    # Fallback if already cached in a standard location
    MODEL_DIR = Path.home() / ".cache/kagglehub/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"

# Check/download Perch ONNX model
ONNX_PERCH_PATH = Path("")
for pat in ["**/perch_v2_no_dft*.onnx", "**/perch_v2*.onnx"]:
    hits = sorted(BASE.glob(pat))
    if hits:
        ONNX_PERCH_PATH = hits[0]
        break

if not ONNX_PERCH_PATH.is_file():
    print("Checking/Downloading Perch ONNX model via kagglehub...")
    try:
        ONNX_DIR = Path(kagglehub.dataset_download("rishikeshjani/perch-onnx-for-birdclef-2026"))
        ONNX_PERCH_PATH = next(ONNX_DIR.glob("**/perch_v2_no_dft*.onnx"),
                               next(ONNX_DIR.glob("**/perch_v2*.onnx"), Path("")))
    except Exception as e:
        print(f"Warning: Failed to download Perch ONNX dataset: {e}")

if ONNX_PERCH_PATH.is_file():
    print(f"Found Perch ONNX model: {ONNX_PERCH_PATH}")
else:
    print("Warning: Perch ONNX model not found. Will fallback to TF SavedModel if available.")

# Load TF conditionally if ONNX is missing or if we fall back
USE_ONNX = ONNX_PERCH_PATH.is_file()
if not USE_ONNX:
    import tensorflow as tf
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    try:
        tf.config.set_visible_devices([], "GPU")  # force CPU
    except:
        pass

DEVICE = "cpu"

def seed_everything(s=42):
    random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    torch.manual_seed(s)

seed_everything(42)

_WALL = time.time()
SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12

# ── load trained metadata ──────────────────────────────────────────
META_FILE = TRAINED / "meta.json"
assert META_FILE.exists(), f"Could not find metadata file {META_FILE}"
META = json.load(open(META_FILE))

PRIMARY_LABELS = META["primary_labels"]
N_CLASSES = META["n_classes"]
N_SITES_CAP = META["n_sites"]
ENSEMBLE_W = META["ensemble_w"]
ALPHA_BLEND = META["alpha_blend"]
CORRECTION_WEIGHT = META["correction_weight"]
SITE2I = META["site2i"]
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

# sanity check columns vs submission
sample_sub_path = BASE / "sample_submission.csv"
assert sample_sub_path.exists(), f"Missing sample_submission.csv at {sample_sub_path}"
sample_sub = pd.read_csv(sample_sub_path)
assert sample_sub.columns[1:].tolist() == PRIMARY_LABELS, "Label schema mismatch vs training!"

taxonomy_path = BASE / "taxonomy.csv"
assert taxonomy_path.exists(), f"Missing taxonomy.csv at {taxonomy_path}"
taxonomy = pd.read_csv(taxonomy_path)

# ── Perch backbone session setup ─────────────────
if USE_ONNX:
    providers = ["CPUExecutionProvider"]
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = 4
    _so.inter_op_num_threads = 4
    _so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    ONNX_SESSION = ort.InferenceSession(str(ONNX_PERCH_PATH), sess_options=_so, providers=providers)
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_OUT_MAP = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
    print("Using ONNX Perch on:", ONNX_SESSION.get_providers())
else:
    assert MODEL_DIR.exists(), f"Perch model directory not found at {MODEL_DIR}"
    birdclassifier = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = birdclassifier.signatures["serving_default"]
    print("Using TF SavedModel Perch (CPU)")

# ── species mapping ────────────────────
bc_labels = (pd.read_csv(MODEL_DIR / "assets" / "labels.csv").reset_index()
             .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}))
NO_LABEL = len(bc_labels)
mapping = taxonomy.merge(bc_labels, on="scientific_name", how="left")
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
lbl2bc = mapping.set_index("primary_label")["bc_index"]
BC_INDICES = np.array([int(lbl2bc.loc[c]) for c in PRIMARY_LABELS], np.int32)
MAPPED_MASK = BC_INDICES != NO_LABEL
MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA = {"Amphibia", "Insecta"}

proxy_map = {}
for _, row in taxonomy[taxonomy["primary_label"].isin([PRIMARY_LABELS[i] for i in UNMAPPED_POS])].iterrows():
    genus = str(row["scientific_name"]).split()[0]
    hits = bc_labels[bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)]
    if len(hits) > 0:
        proxy_map[label_to_idx[row["primary_label"]]] = hits["bc_index"].astype(int).tolist()
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
proxy_map = {i: v for i, v in proxy_map.items() if CLASS_NAME_MAP.get(PRIMARY_LABELS[i]) in PROXY_TAXA}

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

def parse_fname(name):
    m = FNAME_RE.match(name)
    if not m:
        return {"site": "unknown", "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}

import concurrent.futures

def read_60s(path):
    y, _ = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(1)
    return np.pad(y, (0, FILE_SAMPLES - len(y))) if len(y) < FILE_SAMPLES else y[:FILE_SAMPLES]

def run_perch(paths, batch_files=16, verbose=True):
    paths = [Path(p) for p in paths]
    n = len(paths) * N_WINDOWS
    row_ids = np.empty(n, object)
    filenames = np.empty(n, object)
    sites = np.empty(n, object)
    hours = np.zeros(n, np.int16)
    scores = np.zeros((n, N_CLASSES), np.float32)
    embs = np.zeros((n, 1536), np.float32)
    wr = 0
    itr = tqdm(range(0, len(paths), batch_files), desc="Perch") if verbose else range(0, len(paths), batch_files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io:
        nxt = paths[0:batch_files]
        fut = [io.submit(read_60s, p) for p in nxt]
        for start in itr:
            bp = nxt
            bn = len(bp)
            ba = [f.result() for f in fut]
            ns = start + batch_files
            if ns < len(paths):
                nxt = paths[ns:ns+batch_files]
                fut = [io.submit(read_60s, p) for p in nxt]
            x = np.empty((bn * N_WINDOWS, WINDOW_SAMPLES), np.float32)
            br = wr
            for bi, path in enumerate(bp):
                y = ba[bi]
                meta = parse_fname(path.name)
                stem = path.stem
                x[bi*N_WINDOWS:(bi+1)*N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
                row_ids[wr:wr+N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[wr:wr+N_WINDOWS] = path.name
                sites[wr:wr+N_WINDOWS] = meta["site"]
                hours[wr:wr+N_WINDOWS] = meta["hour_utc"]
                wr += N_WINDOWS
            if USE_ONNX:
                outs = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
                logits = outs[ONNX_OUT_MAP["label"]].astype(np.float32)
                emb = outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)
            else:
                out = infer_fn(inputs=tf.convert_to_tensor(x))
                logits = out["label"].numpy().astype(np.float32)
                emb = out["embedding"].numpy().astype(np.float32)
            scores[br:wr, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
            embs[br:wr] = emb
            for pi, bc in proxy_map.items():
                scores[br:wr, pi] = logits[:, np.array(bc, np.int32)].max(1)
            del x, logits, emb, ba
            gc.collect()
    return pd.DataFrame({"row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours}), scores, embs

print("Perch engine + mappings ready")

# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

class VectorizedMLPProbes(nn.Module):
    def __init__(self, pm):
        super().__init__()
        self.valid_classes = sorted(pm.keys())
        V = len(self.valid_classes)
        if V == 0:
            self.weights = nn.ParameterList()
            self.biases = nn.ParameterList()
            self.n_layers = 0
            return
        s = pm[self.valid_classes[0]]
        self.n_layers = len(s.coefs_)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for li in range(self.n_layers):
            W = np.stack([pm[c].coefs_[li] for c in self.valid_classes], 0)
            b = np.stack([pm[c].intercepts_[li] for c in self.valid_classes], 0)
            self.weights.append(nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False))
            self.biases.append(nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=False))

    def forward(self, x):
        h = x
        for i in range(self.n_layers):
            h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
            if i < self.n_layers - 1:
                h = torch.relu(h)
        return h.squeeze(-1)

def apply_mlp_probes_vectorized(emb_test, scores_test, pm, scaler, pca, alpha_blend=0.4):
    if len(pm) == 0:
        return scores_test.copy()
    Z = pca.transform(scaler.transform(emb_test)).astype(np.float32)
    vc = sorted(pm.keys())
    V = len(vc)
    N = len(scores_test)
    raw = scores_test[:, vc].T
    nf = N // N_WINDOWS
    rv = raw.reshape(V, nf, N_WINDOWS)
    prev = np.concatenate([rv[:, :, :1], rv[:, :, :-1]], 2).reshape(V, N)
    nxt = np.concatenate([rv[:, :, 1:], rv[:, :, -1:]], 2).reshape(V, N)
    mean = np.repeat(rv.mean(2), N_WINDOWS, 1)
    mx = np.repeat(rv.max(2), N_WINDOWS, 1)
    std = np.repeat(rv.std(2), N_WINDOWS, 1)
    sfx = np.stack([raw, prev, nxt, mean, mx, std], -1).astype(np.float32)
    Ze = np.broadcast_to(Z, (V, N, Z.shape[1]))
    X = np.concatenate([Ze.astype(np.float32), sfx], -1)
    vp = VectorizedMLPProbes(pm).eval()
    with torch.no_grad():
        preds = vp(torch.tensor(X)).numpy()
    out = scores_test.copy()
    out[:, vc] = (1.0 - alpha_blend) * scores_test[:, vc] + alpha_blend * preds.T
    return out

def apply_prior(scores, sites, hours, tables, lambda_prior=0.4):
    eps = 1e-4
    n = len(scores)
    out = scores.copy()
    p = np.tile(tables["global_p"], (n, 1))
    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j = tables["hour_to_i"][h]
            nh = tables["hour_n"][j]
            w = nh / (nh + 8.0)
            p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]
    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j = tables["site_to_i"][s]
            ns = tables["site_n"][j]
            w = ns / (ns + 8.0)
            p[i] = w * tables["site_p"][j] + (1 - w) * p[i]
    for i, (s, h) in enumerate(zip(sites, hours)):
        key = (str(s), int(h))
        if key in tables["sh_to_i"]:
            j = tables["sh_to_i"][key]
            nsh = tables["sh_n"][j]
            w = nsh / (nsh + 4.0)
            p[i] = w * tables["sh_p"][j] + (1 - w) * p[i]
    p = np.clip(p, eps, 1 - eps)
    out += lambda_prior * (np.log(p) - np.log1p(-p))
    return out.astype(np.float32)

def file_confidence_scale(probs, n_windows=12, top_k=2, power=0.4):
    N, C = probs.shape
    v = probs.reshape(-1, n_windows, C)
    sv = np.sort(v, 1)
    tk = sv[:, -top_k:, :].mean(1, keepdims=True)
    return (v * np.power(tk, power)).reshape(N, C)

def rank_aware_scaling(probs, n_windows=12, power=0.4):
    N, C = probs.shape
    v = probs.reshape(-1, n_windows, C)
    fm = v.max(1, keepdims=True)
    return (v * np.power(fm, power)).reshape(N, C)

def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    N, C = probs.shape
    r = probs.copy()
    v = probs.reshape(-1, n_windows, C)
    o = r.reshape(-1, n_windows, C)
    for t in range(n_windows):
        conf = v[:, t, :].max(-1, keepdims=True)
        a = base_alpha * (1.0 - conf)
        if t == 0:
            na = (v[:, t, :] + v[:, t + 1, :]) / 2.0
        elif t == n_windows - 1:
            na = (v[:, t - 1, :] + v[:, t, :]) / 2.0
        else:
            na = (v[:, t - 1, :] + v[:, t + 1, :]) / 2.0
        o[:, t, :] = (1.0 - a) * v[:, t, :] + a * na
    return r

def apply_per_class_thresholds(scores, thr):
    C = scores.shape[1]
    sc = np.copy(scores)
    for c in range(C):
        t = thr[c]
        ab = scores[:, c] > t
        sc[ab, c] = 0.5 + 0.5 * (scores[ab, c] - t) / (1 - t + 1e-8)
        sc[~ab, c] = 0.5 * scores[~ab, c] / (t + 1e-8)
    return np.clip(sc, 0.0, 1.0)

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_sz, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, -1)
        x_conv = F.silu(self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2))
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        h = torch.zeros(B_sz, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))
        return torch.stack(ys, 1) + x * self.D[None, None, :]

class LightProtoSSM(nn.Module):
    def __init__(self, d_input=1536, d_model=128, d_state=16, n_classes=234, n_windows=12, dropout=0.15,
                 n_sites=20, meta_dim=16, use_cross_attn=True, cross_attn_heads=2):
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn
        self.input_proj = nn.Sequential(nn.Linear(d_input, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.ssm_fwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_bwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(2)])
        self.ssm_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.drop = nn.Dropout(dropout)
        if use_cross_attn:
            self.cross_attn = nn.ModuleList([nn.MultiheadAttention(d_model, cross_attn_heads, dropout=dropout, batch_first=True) for _ in range(2)])
            self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], -1))
            h = h + meta[:, None, :]
        for i, (fwd, bwd, merge, norm) in enumerate(zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm)):
            res = h
            hf = fwd(h)
            hb = bwd(h.flip(1)).flip(1)
            h = self.drop(merge(torch.cat([hf, hb], -1)))
            h = norm(h + res)
            if self.use_cross_attn:
                a, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + a)
        hn = F.normalize(h, dim=-1)
        pn = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(hn, pn.T) * F.softplus(self.proto_temp) + self.class_bias[None, None, :]
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            return alpha * sim + (1 - alpha) * perch_logits
        return sim

class ResidualSSM(nn.Module):
    def __init__(self, d_input=1536, d_scores=234, d_model=64, d_state=8, n_classes=234, n_windows=12, dropout=0.1, n_sites=20, meta_dim=8):
        super().__init__()
        self.n_classes = n_classes
        self.input_proj = nn.Sequential(nn.Linear(d_input + d_scores, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm_drop = nn.Dropout(dropout)
        self.output_head = nn.Linear(d_model, n_classes)

    def forward(self, emb, first_pass, site_ids=None, hours=None):
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], -1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat([self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1)), self.hour_emb(hours.clamp(0, 23))], -1))
            h = h + meta.unsqueeze(1)
        res = h
        hf = self.ssm_fwd(h)
        hb = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.ssm_drop(self.ssm_merge(torch.cat([hf, hb], -1)))
        h = self.ssm_norm(h + res)
        return self.output_head(h)

# ── load trained artifacts ────────────────────────────────────────────────────
proto_model = LightProtoSSM(n_classes=N_CLASSES, n_sites=N_SITES_CAP, use_cross_attn=True, cross_attn_heads=2)
proto_model.load_state_dict(torch.load(TRAINED / "proto_ssm.pt", map_location="cpu"))
proto_model.eval()

res_model = ResidualSSM(n_classes=N_CLASSES)
res_model.load_state_dict(torch.load(TRAINED / "residual_ssm.pt", map_location="cpu"))
res_model.eval()

probe_models = joblib.load(TRAINED / "mlp_probes.joblib")
emb_scaler = joblib.load(TRAINED / "emb_scaler.joblib")
emb_pca = joblib.load(TRAINED / "emb_pca.joblib")
prior_tables = joblib.load(TRAINED / "prior_tables.joblib")
PER_CLASS_THRESHOLDS = np.load(TRAINED / "thresholds.npy")
temperatures = np.load(TRAINED / "temperatures.npy")
correction_weight = CORRECTION_WEIGHT
print("Loaded all trained artifacts successfully")

# ─────────────────────────────────────────────────────────────────────────────
# 1. PROTOSSM BRANCH INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
IS_DRY_RUN = len(test_paths) == 0

if IS_DRY_RUN:
    print("No hidden test files found — dry-run on 20 train files")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:20]
else:
    print(f"Hidden test files: {len(test_paths)}")

assert len(test_paths) > 0, "No audio files found for inference!"

meta_te, sc_te, emb_te = run_perch(test_paths, batch_files=4, verbose=True)
print(f"Test scores shape: {sc_te.shape}")

nft = len(sc_te) // N_WINDOWS
emb_te_f = emb_te.reshape(nft, N_WINDOWS, -1)
sc_te_f = sc_te.reshape(nft, N_WINDOWS, -1)
te_fn = meta_te.drop_duplicates("filename")["filename"].tolist()
te_site = np.array([min(SITE2I.get(meta_te.loc[meta_te["filename"] == fn, "site"].iloc[0], 0), N_SITES_CAP - 1) for fn in te_fn], np.int64)
te_hour = np.array([int(meta_te.loc[meta_te["filename"] == fn, "hour_utc"].iloc[0]) % 24 for fn in te_fn], np.int64)

with torch.no_grad():
    proto_out = proto_model(torch.tensor(emb_te_f, dtype=torch.float32), torch.tensor(sc_te_f, dtype=torch.float32),
                            site_ids=torch.tensor(te_site, dtype=torch.long), hours=torch.tensor(te_hour, dtype=torch.long)).numpy()
proto_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)

sc_te_adj = apply_prior(sc_te, meta_te["site"].to_numpy(), meta_te["hour_utc"].to_numpy(), prior_tables, 0.4)
sc_te_adj = apply_mlp_probes_vectorized(emb_te, sc_te_adj, probe_models, emb_scaler, emb_pca, ALPHA_BLEND)
first_pass = ENSEMBLE_W * proto_flat + (1.0 - ENSEMBLE_W) * sc_te_adj

fp_f = first_pass.reshape(nft, N_WINDOWS, -1)
with torch.no_grad():
    corr = res_model(torch.tensor(emb_te_f, dtype=torch.float32), torch.tensor(fp_f, dtype=torch.float32),
                     site_ids=torch.tensor(te_site, dtype=torch.long), hours=torch.tensor(te_hour, dtype=torch.long)).numpy()
final = first_pass + correction_weight * corr.reshape(-1, N_CLASSES).astype(np.float32)
final = final / temperatures[None, :]
probs = sigmoid(final)
probs = file_confidence_scale(probs, N_WINDOWS, 2, 0.4)
probs = rank_aware_scaling(probs, N_WINDOWS, 0.4)
probs = adaptive_delta_smooth(probs, N_WINDOWS, 0.20)
probs = np.clip(probs, 0.0, 1.0)
probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)

sub = pd.DataFrame(probs.astype(np.float32), columns=PRIMARY_LABELS)
sub.insert(0, "row_id", meta_te["row_id"].values)
sub.to_csv("submission_protossm.csv", index=False)
del proto_model, res_model
gc.collect()
print("ProtoSSM branch done.", f"{(time.time() - _WALL) / 60:.1f} min")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DISTILLED SED BRANCH INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
N_MELS_SED = 256
N_FFT_SED = 2048
HOP_SED = 512
FMIN_SED = 20
FMAX_SED = 16000
TOP_DB_SED = 80

# Try to find SED fold models locally first
sed_dir = None
hits = sorted(BASE.rglob("sed_fold0.onnx"))
if hits:
    sed_dir = hits[0].parent
else:
    print("Downloading distilled SED models via kagglehub...")
    try:
        sed_dir = Path(kagglehub.dataset_download("tuckerarrants/bc2026-distilled-sed-public"))
    except Exception as e:
        print(f"Warning: Failed to download SED models via kagglehub: {e}")

if sed_dir and sed_dir.exists():
    sed_fold_paths = sorted(sed_dir.glob("sed_fold*.onnx"), key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)))
else:
    sed_fold_paths = []

def make_sed_session(p):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])

def audio_to_mel(chunks):
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED, n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0)
        s = librosa.power_to_db(s, top_db=TOP_DB_SED)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)

def file_to_sed_chunks(path):
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(1)
    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    n = 60 * SR
    y = np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]
    return y.reshape(N_WINDOWS, WINDOW_SAMPLES), np.arange(1, N_WINDOWS + 1) * WINDOW_SEC

def sigmoid_sed(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)

if not sed_fold_paths:
    print("SED models not found — skipping SED branch (zero submission fallback)")
    sed_sub = pd.read_csv("submission_protossm.csv")
    for c in PRIMARY_LABELS:
        sed_sub[c] = 0.0
    sed_sub.to_csv("submission_sed.csv", index=False)
else:
    sed_sessions = [make_sed_session(p) for p in sed_fold_paths]
    print("SED folds:", [p.name for p in sed_fold_paths])
    sed_rows, sed_preds = [], []
    for i, path in enumerate(test_paths, 1):
        chunks, ends = file_to_sed_chunks(path)
        mel = audio_to_mel(chunks)
        ps = np.zeros((len(chunks), N_CLASSES), np.float32)
        for sess in sed_sessions:
            outs = sess.run(None, {sess.get_inputs()[0].name: mel})
            ps += 0.5 * sigmoid_sed(outs[0]) + 0.5 * sigmoid_sed(outs[1].max(1))
        pm = ps / len(sed_sessions)
        if len(pm) > 1:
            pm = gaussian_filter1d(pm, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        sed_rows.extend([f"{path.stem}_{int(t)}" for t in ends])
        sed_preds.append(pm)
        if i == 1 or i % 50 == 0 or i == len(test_paths):
            print(f"SED: {i}/{len(test_paths)}")
    sed_sub = pd.DataFrame(np.clip(np.concatenate(sed_preds, 0), 0.0, 1.0), columns=PRIMARY_LABELS)
    sed_sub.insert(0, "row_id", sed_rows)
    sed_sub.to_csv("submission_sed.csv", index=False)
    print("SED done:", sed_sub.shape)

# ─────────────────────────────────────────────────────────────────────────────
# 3. BIRDNET BRANCH INFERENCE (graceful skip fallback if offline)
# ─────────────────────────────────────────────────────────────────────────────
BIRDNET_SR = 48_000
BIRDNET_CHUNK_SEC = 3
BIRDNET_CHUNK_SAMPLES = BIRDNET_SR * BIRDNET_CHUNK_SEC

# Try to find BirdNET model/labels locally first
_bn_model_path, _bn_labels_path = None, None
hits = sorted(BASE.rglob("birdnet_global_6k_v2.4_model_fp32*.tflite"))
if hits:
    _bn_model_path = hits[0]
    lbl_hits = sorted(hits[0].parent.rglob("*Labels*.txt"))
    if lbl_hits:
        _bn_labels_path = lbl_hits[0]
else:
    print("Attempting to download BirdNET model via kagglehub...")
    try:
        bn_dir = Path(kagglehub.model_download("shadiakiki1/birdnet-analyzer/tfLite/birdnet_global_6k_v2.4_model_fp32-1"))
        _bn_model_path = next(bn_dir.glob("**/birdnet_global_6k_v2.4_model_fp32*.tflite"),
                              next(bn_dir.glob("**/BirdNET_GLOBAL_6K_V2.4_Model_FP32*.tflite"), None))
        _bn_labels_path = next(bn_dir.glob("**/BirdNET_GLOBAL_6K_V2.4_Labels.txt"),
                               next(bn_dir.glob("**/birdnet*labels*.txt"), None))
    except Exception as e:
        print(f"Warning: Failed to download BirdNET model/labels via kagglehub: {e}")

if _bn_model_path is None:
    USE_BIRDNET = False
    BN_TO_COMP = {}
    BN_PROXY = {}
    _bn_labels_raw = []
    print("BirdNET not found — 60/40 fallback")
else:
    _TFLiteInterp = None
    try:
        from tflite_runtime.interpreter import Interpreter as _TFLiteInterp
    except ImportError:
        import sys
        if not (sys.platform == "darwin" and sys.version_info >= (3, 13)):
            try:
                from tensorflow.lite.python.interpreter import Interpreter as _TFLiteInterp
            except ImportError:
                pass
            except Exception:
                pass
    
    if _TFLiteInterp is None:
        USE_BIRDNET = False
        BN_TO_COMP = {}
        BN_PROXY = {}
        _bn_labels_raw = []
        print("BirdNET interpreter could not be imported — 60/40 fallback")
    else:
        USE_BIRDNET = True
        print("BirdNET model:", _bn_model_path.name)
        _bn_interp = _TFLiteInterp(model_path=str(_bn_model_path), num_threads=4)
        _bn_interp.allocate_tensors()
        _bn_in = _bn_interp.get_input_details()[0]
        _bn_out = _bn_interp.get_output_details()
        _bn_logit_idx = _bn_out[-1]["index"]
        _bn_labels_raw = [l.strip() for l in _bn_labels_path.read_text().splitlines() if l.strip()] if _bn_labels_path else []
        _bn_sci = [lbl.split("_", 1)[0].strip() for lbl in _bn_labels_raw]
        _tax_sci = taxonomy.set_index("scientific_name")["primary_label"].to_dict()
        BN_TO_COMP = {}
        for bn_i, sci in enumerate(_bn_sci):
            if sci in _tax_sci and _tax_sci[sci] in label_to_idx:
                BN_TO_COMP[bn_i] = label_to_idx[_tax_sci[sci]]
        _mapped = set(BN_TO_COMP.values())
        BN_PROXY = {}
        for ci, primary in enumerate(PRIMARY_LABELS):
            if ci in _mapped:
                continue
            row = taxonomy[taxonomy["primary_label"] == primary]
            if row.empty:
                continue
            genus = str(row.iloc[0]["scientific_name"]).split()[0]
            idxs = [i for i, s in enumerate(_bn_sci) if s.startswith(genus + " ")]
            if idxs:
                BN_PROXY[ci] = idxs
        print(f"BirdNET map: {len(BN_TO_COMP)} direct + {len(BN_PROXY)} proxy")

_N_BN = 20
_win_to_chunks = [[j for j in range(_N_BN) if 3 * j < (w + 1) * 5 and 3 * (j + 1) > w * 5] for w in range(N_WINDOWS)]

def run_birdnet(paths, verbose=True):
    if not USE_BIRDNET or not _bn_labels_raw:
        return None, None
    paths = [Path(p) for p in paths]
    n = len(paths) * N_WINDOWS
    row_ids = np.empty(n, object)
    filenames = np.empty(n, object)
    scores = np.zeros((n, N_CLASSES), np.float32)
    wr = 0
    for path in (tqdm(paths, desc="BirdNET") if verbose else paths):
        y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(1)
        if sr0 != BIRDNET_SR:
            y = librosa.resample(y, orig_sr=sr0, target_sr=BIRDNET_SR)
        tgt = 60 * BIRDNET_SR
        y = np.pad(y, (0, tgt - len(y))) if len(y) < tgt else y[:tgt]
        chunks = y.reshape(_N_BN, BIRDNET_CHUNK_SAMPLES)
        cp = np.zeros((_N_BN, len(_bn_labels_raw)), np.float32)
        for j, ch in enumerate(chunks):
            _bn_interp.set_tensor(_bn_in["index"], ch[None, :].astype(np.float32))
            _bn_interp.invoke()
            lg = _bn_interp.get_tensor(_bn_logit_idx)[0]
            cp[j] = 1.0 / (1.0 + np.exp(-np.clip(lg, -50, 50)))
        for w, cl in enumerate(_win_to_chunks):
            wp = cp[cl].max(0)
            r = wr + w
            row_ids[r] = f"{path.stem}_{(w + 1) * 5}"
            filenames[r] = path.name
            for bn_i, ci in BN_TO_COMP.items():
                if wp[bn_i] > scores[r, ci]:
                    scores[r, ci] = wp[bn_i]
            for ci, bi in BN_PROXY.items():
                v = wp[bi].max()
                if v > scores[r, ci]:
                    scores[r, ci] = v
        wr += N_WINDOWS
    return pd.DataFrame({"row_id": row_ids[:wr], "filename": filenames[:wr]}), scores[:wr]

if USE_BIRDNET:
    _m, _s = run_birdnet(test_paths, verbose=True)
    _sv = _s.reshape(len(_s) // N_WINDOWS, N_WINDOWS, N_CLASSES)
    for fi in range(len(_sv)):
        _sv[fi] = gaussian_filter1d(_sv[fi], sigma=0.65, axis=0, mode="nearest")
    _s = _sv.reshape(-1, N_CLASSES)
    _bs = pd.DataFrame(np.clip(_s, 0.0, 1.0), columns=PRIMARY_LABELS)
    _bs.insert(0, "row_id", _m["row_id"].values)
    _bs.to_csv("submission_birdnet.csv", index=False)
    print("BirdNET coverage:", (_s > 0.01).any(0).sum())
else:
    _d = pd.read_csv("submission_protossm.csv")
    for c in PRIMARY_LABELS:
        _d[c] = 0.0
    _d.to_csv("submission_birdnet.csv", index=False)
    print("BirdNET zero submission generated (fallback)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. RANK BLEND + GATES BLENDING AND MIRROR PAIRS
# ─────────────────────────────────────────────────────────────────────────────
EPS = 1e-5
df_proto = pd.read_csv("submission_protossm.csv")
df_sed = pd.read_csv("submission_sed.csv")
cols = [c for c in df_proto.columns if c != "row_id"]

df_sed = df_sed.set_index("row_id").loc[df_proto["row_id"]].reset_index()
p_proto = np.clip(df_proto[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
p_sed = np.clip(df_sed[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
rank_proto = pd.DataFrame(p_proto).rank(axis=0, pct=True).to_numpy(np.float32)
rank_sed = pd.DataFrame(p_sed).rank(axis=0, pct=True).to_numpy(np.float32)

try:
    df_bn = pd.read_csv("submission_birdnet.csv").set_index("row_id").loc[df_proto["row_id"]].reset_index()
    p_bn = np.clip(df_bn[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
    rank_bn = pd.DataFrame(p_bn).rank(axis=0, pct=True).to_numpy(np.float32) if (p_bn > 0.01).any() else None
except Exception as e:
    rank_bn = None
    print("BirdNET load failed:", e)

pred = (rank_proto * 0.50) + (rank_sed * 0.30) + (rank_bn * 0.20) if rank_bn is not None else (rank_proto * 0.60) + (rank_sed * 0.40)
print("3-way blend" if rank_bn is not None else "2-way blend")

row_ids = df_proto["row_id"].astype(str).to_numpy()
file_ids = np.array(["_".join(r.split("_")[:-1]) for r in row_ids])
fake_only = (p_proto > 0.50) & (p_sed < 0.05)
pred = np.where(fake_only, 0.92 * pred + 0.08 * rank_proto, pred)

offs = np.arange(-3, 4, dtype=np.float32)
pk = (1.0 + (offs / 1.20) ** 2 / 2.0) ** (-1.5)
pk = (pk / pk.sum()).astype(np.float32)
pa = p_proto.copy()
for fid in pd.unique(file_ids):
    m = file_ids == fid
    x = p_proto[m]
    if len(x) > 1:
        xp = np.pad(x, ((3, 3), (0, 0)), mode="edge")
        pa[m] = sum(pk[i] * xp[i:i+len(x)] for i in range(7))

xctx = pd.DataFrame(pa).rank(axis=0, pct=True).to_numpy(np.float32)
proto_cont = (xctx > 0.88) & (rank_proto > 0.75) & (p_sed < 0.12) & (~fake_only)
pred = np.where(proto_cont, 0.85 * pred + 0.15 * np.maximum(rank_proto, xctx), pred)
sed_only = (rank_sed > 0.95) & (rank_proto < 0.80) & (~fake_only) & (~proto_cont)
pred = np.where(sed_only, 0.88 * pred + 0.12 * rank_sed, pred)

if rank_bn is not None:
    bn_only = (rank_bn > 0.95) & (rank_proto < 0.75) & (rank_sed < 0.80) & (~fake_only) & (~proto_cont) & (~sed_only)
    pred = np.where(bn_only, 0.90 * pred + 0.10 * rank_bn, pred)

sub = df_proto.copy()
sub[cols] = pred.astype(np.float32)

MIRROR_PAIRS = (
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23")
)
c2i = {l: i for i, l in enumerate(cols)}
mc = 0
for g in MIRROR_PAIRS:
    vi = [c2i[s] for s in g if s in c2i]
    if len(vi) >= 2:
        gm = sub[cols].iloc[:, vi].max(axis=1).to_numpy(np.float32)
        for idx in vi:
            sub.iloc[:, idx + 1] = gm
        mc += len(vi)
print("Mirrored cols:", mc)

try:
    tax_df = pd.read_csv(BASE / "taxonomy.csv").set_index("primary_label")
    rare = {"Amphibia", "Mammalia", "Reptilia"}
    rc = 0
    for ci, sp in enumerate(cols):
        if sp in tax_df.index and tax_df.loc[sp, "class_name"] in rare:
            ci1 = ci + 1
            vals = sub.iloc[:, ci1].to_numpy(np.float32)
            thr = vals.mean() + 0.05
            sub.iloc[:, ci1] = np.where(vals < thr, vals * 0.9, vals)
            rc += 1
    print("Rare thresholded:", rc)
except Exception as e:
    print("Rare species skip:", e)

if IS_DRY_RUN:
    print("Dry-run alignment with sample_submission")
    sp = pd.read_csv(BASE / "sample_submission.csv")
    tmpl = sub[cols].mean(axis=0).astype(np.float32)
    sub = sp.copy()
    for l in cols:
        sub[l] = tmpl[l]

sub.to_csv("submission.csv", index=False)
print("Saved final submission.csv", sub.shape, "Ready!")
print(f"Total time elapsed: {(time.time() - _WALL) / 60:.1f} minutes")
