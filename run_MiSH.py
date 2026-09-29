#!/usr/bin/env python
import os
import subprocess

# Configuration
root_dir = "/root/autodl-tmp/Datasets"
tau = 0.015
margin = 0.1
noisy_rate=0.0  #0.0 0.2 0.5 0.8
# noisy_file = "RDE-base/noiseindex"
select_ratio = 0.3
loss = "TAL"
DATASET_NAME = "RSTPReid"
# CUHK-PEDES ICFG-PEDES RSTPReid

# ---- MiSH: Minor-Subspace Householder Reflection Adaptation ----
freeze_clip = True   # freeze CLIP, train only HRA + TSE + LayerNorm
hra_r = 32           # number of Householder reflections per adapted linear
hra_where = "mlp"   # 'mlp' only (attn unsupported: MHA uses weights functionally)
hra_lambda = 1e-3   # orthogonality regularisation strength

# ---- MSC: Minor-Subspace Confinement ----
# Protect the leading principal input directions covering this fraction of each
# layer's spectral energy (CLIP's pretrained image-text alignment); adapt only in
# the low-energy minor subspace. 0.0 -> disabled (recovers vanilla HRA).
# Set manually with DATASET_NAME: CUHK-PEDES=0.5, ICFG-PEDES=0.2, RSTPReid=0.9.
hra_subspace = 0.9

# ---- DGM: Directional Gain Modulation ----
# In the MSC-protected minor subspace, the fixed reflection coefficient 2 becomes
# a per-axis learnable scalar beta_j (H_j = I - beta_j u_j u_j^T). span(V_p) stays
# exactly protected (u_j _|_ V_p => H_j v = v), while the minor-axis gain 1-beta_j
# becomes learnable -> reflections can amplify/attenuate minor directions, not only
# rotate them. beta init=2 makes DGM a strict superset of HRA+MSC.
dgm = True
dgm_lambda = 1e-4    # pulls beta back toward 2; 0 disables the regulariser

# Set environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Build command
cmd = [
    "python", "train.py",
    "--name", "MiSH",
    "--img_aug",
    "--txt_aug",
    "--batch_size", "64",
    "--select_ratio", str(select_ratio),
    "--tau", str(tau),
    "--noisy_rate", str(noisy_rate),
    "--root_dir", root_dir,
    "--output_dir", "run_logs",
    "--margin", str(margin),
    "--dataset_name", DATASET_NAME,
    "--loss_names", f"{loss}+sr{select_ratio}_tau{tau}_margin{margin}_nr{noisy_rate}",
    "--num_epoch", "60",
    "--hra_r", str(hra_r),
    "--hra_where", hra_where,
    "--hra_lambda", str(hra_lambda),
]
if freeze_clip:
    cmd.append("--freeze_clip")
if hra_subspace and hra_subspace > 0:
    cmd += ["--hra_subspace", str(hra_subspace)]
if dgm:
    cmd.append("--dgm")
    cmd += ["--dgm_lambda", str(dgm_lambda)]

# Run command
subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
