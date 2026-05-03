# MLPerf Inference v6.0 — Exercise Log

**Hardware:** B300-SXM-270GBx8  
**OS:** Ubuntu 24.04.3 LTS  
**Repo:** https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/NVIDIA

---

## Reference Documents

| Document | URL |
|---|---|
| NVIDIA MLPerf v6.0 README | https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/NVIDIA |
| Llama2-70b Benchmark README | https://github.com/mlcommons/inference_results_v6.0/blob/main/closed/NVIDIA/code/llama2-70b/tensorrt/README.md |
| Scaleout README | https://github.com/mlcommons/inference_results_v6.0/blob/main/closed/NVIDIA/scaleout/README.md |
| Scaleout REPRODUCE.md | https://github.com/mlcommons/inference_results_v6.0/blob/main/closed/NVIDIA/scaleout/REPRODUCE.md |
| MLCommons Llama2-70b Dataset README | https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md |

---

## Environment

| Component | Version / Detail |
|---|---|
| NVIDIA Driver | 580.126.20 |
| CUDA | 13.0 |
| GPUs | 8x NVIDIA B300 SXM6 275GB |
| Slurm | 23.11.4 |
| Enroot | 4.1.0 |
| Pyxis | 0.21.0 |

---

## Step 1 — Install Slurm

```bash
sudo apt install -y slurmd slurmctld munge
```

Get node hardware info for slurm.conf:

```bash
sudo slurmd -C
# NodeName=ark CPUs=256 Boards=1 SocketsPerBoard=2 CoresPerSocket=64 ThreadsPerCore=2 RealMemory=2061780
```

---

## Step 2 — Configure Slurm

Create required directories:

```bash
sudo mkdir -p /var/log/slurm /var/spool/slurm/ctld /var/spool/slurmd
sudo chown -R slurm:slurm /var/spool/slurm/ /var/log/slurm
sudo chmod 755 /var/spool/slurm/ctld
```

Write `/etc/slurm/slurm.conf`:

```ini
# Cluster identity
SlurmUser=slurm
ClusterName=ark-cluster
SlurmctldHost=ark

# Authentication
AuthType=auth/munge

# Prolog/Epilog for GPU cleanup
PrologFlags=Alloc

# Scheduling
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

# Logging
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmdPidFile=/var/run/slurmd.pid

# Spool directories
StateSaveLocation=/var/spool/slurm/ctld
SlurmdSpoolDir=/var/spool/slurmd

# Timeouts
SlurmctldTimeout=300
SlurmdTimeout=300
InactiveLimit=0
MinJobAge=300
KillWait=30
Waittime=0

# GPU support via GRES
GresTypes=gpu

# Node definition
NodeName=ark \
    CPUs=256 \
    Boards=1 \
    SocketsPerBoard=2 \
    CoresPerSocket=64 \
    ThreadsPerCore=2 \
    RealMemory=2061780 \
    Gres=gpu:b300:8 \
    State=UNKNOWN

# Partition - named "b300" to match REPRODUCE.md
PartitionName=b300 \
    Nodes=ark \
    Default=YES \
    MaxTime=INFINITE \
    State=UP
```

Write `/etc/slurm/gres.conf`:

```ini
Name=gpu Type=b300 File=/dev/nvidia0
Name=gpu Type=b300 File=/dev/nvidia1
Name=gpu Type=b300 File=/dev/nvidia2
Name=gpu Type=b300 File=/dev/nvidia3
Name=gpu Type=b300 File=/dev/nvidia4
Name=gpu Type=b300 File=/dev/nvidia5
Name=gpu Type=b300 File=/dev/nvidia6
Name=gpu Type=b300 File=/dev/nvidia7
```

Start services:

```bash
sudo systemctl enable --now munge
sudo systemctl enable --now slurmctld
sudo systemctl enable --now slurmd
sinfo
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# b300*        up   infinite      1   idle ark
```

> **Note:** PMIx warnings (`mpi/pmix_v5: can not load PMIx library`) are harmless. MLPerf uses NCCL, not MPI.

> **Note:** `SlurmUser=slurm` must be explicitly set in `slurm.conf`, otherwise slurmd throws "Security violation" errors for uid 64030.

---

## Step 3 — Install Enroot

```bash
wget https://github.com/NVIDIA/enroot/releases/download/v4.1.0/enroot_4.1.0-1_amd64.deb -P /tmp/
wget https://github.com/NVIDIA/enroot/releases/download/v4.1.0/enroot+caps_4.1.0-1_amd64.deb -P /tmp/
sudo apt install -y /tmp/enroot_4.1.0-1_amd64.deb /tmp/enroot+caps_4.1.0-1_amd64.deb

enroot version
# 4.1.0
```

> **Note:** Install both `enroot` and `enroot+caps` — the `+caps` package provides Linux capabilities needed for container operations.

> **Note:** Download to `/tmp/` to avoid `_apt` permission warning during install.

### Enroot Configuration (Required for MLPerf sqsh builds)

Enroot does **not** inherit the host environment like Docker — it starts fresh by design (HPC reproducibility). All env vars must be explicitly passed. Several system-level fixes are required:

**1. Create runtime directory:**
```bash
sudo mkdir -p /run/enroot
sudo chmod 777 /run/enroot
sudo mkdir -p /run/enroot/user/$(id -u)
sudo chown -R $USER:$USER /run/enroot/user/$(id -u)
```

> **Note:** `/run/enroot` is a tmpfs path wiped on every reboot. Make it auto-recreate persistently:
```bash
sudo tee /etc/tmpfiles.d/enroot.conf << 'EOF'
d /run/enroot 0777 root root -
EOF
```
> After adding this file, verify it works after next reboot with `ls /run/enroot`. If missing, run the `mkdir` commands manually as a workaround.

**2. Enable writable rootfs (system-wide, required for Pyxis/slurmd context):**
```bash
# User-level config (for direct enroot use)
mkdir -p ~/.config/enroot
cat > ~/.config/enroot/enroot.conf << 'EOF'
ENROOT_ROOTFS_WRITABLE=yes
ENROOT_RUNTIME_PATH=/run/enroot/user/$(id -u)
ENROOT_CACHE_PATH=${HOME}/.cache/enroot
ENROOT_DATA_PATH=${HOME}/.local/share/enroot
ENROOT_TEMP_PATH=/tmp
EOF

# System-wide config (required — slurmd runs as different user and won't read ~/.config)
sudo tee -a /etc/enroot/enroot.conf << 'EOF'
ENROOT_ROOTFS_WRITABLE=yes
EOF
```

> **Note:** The user-level config alone is NOT enough — Pyxis launches containers via `slurmd` which runs as the `slurm` user. The system-wide `/etc/enroot/enroot.conf` must also have `ENROOT_ROOTFS_WRITABLE=yes`.

**3. Disable AppArmor unprivileged namespace restriction (Ubuntu 24.04):**
```bash
# Disable (required for enroot overlay filesystem)
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0

# Verify
sysctl kernel.apparmor_restrict_unprivileged_userns
# kernel.apparmor_restrict_unprivileged_userns = 0

# Make persistent across reboots
echo 'kernel.apparmor_restrict_unprivileged_userns=0' | sudo tee /etc/sysctl.d/99-enroot.conf
```

> **Note:** Ubuntu 24.04 enables this restriction by default. Without disabling it, enroot cannot create writable overlay filesystems and container builds fail with `Read-only file system` errors.

> **Note:** `sysctl.d` settings ARE persistent across reboots. `tmpfiles.d` for `/run/enroot` may not trigger correctly — always verify after reboot.

---

## Step 4 — Build and Install Pyxis

Pyxis has no prebuilt `.deb` — must build from source against your Slurm version.

```bash
# Install build dependencies
sudo apt install -y libslurm-dev build-essential devscripts debhelper

# Clone into a separate directory (avoid conflict with MLPerf's pyxis/ folder)
cd ~
git clone --depth 1 --branch v0.21.0 https://github.com/NVIDIA/pyxis.git pyxis-build
cd pyxis-build

# Build
make orig
make deb

# Install
sudo dpkg -i ~/nvslurm-plugin-pyxis_0.21.0-1_amd64.deb
```

Configure Slurm to load Pyxis:

```bash
# Create plugstack.conf
sudo tee /etc/slurm/plugstack.conf << 'EOF'
include /etc/slurm/plugstack.conf.d/*.conf
EOF

# Link pyxis config
sudo mkdir -p /etc/slurm/plugstack.conf.d
sudo ln -s /usr/share/pyxis/pyxis.conf /etc/slurm/plugstack.conf.d/pyxis.conf

# Restart slurmd
sudo systemctl restart slurmd

# Verify
srun --help | grep container-image
#       --container-image=[USER@][REGISTRY#]IMAGE[:TAG]|PATH
```

---

## Step 5 — Sanity Test (Slurm + Pyxis + Enroot)

```bash
srun --partition=b300 --container-image=ubuntu:22.04 echo "Pyxis container works!"
# pyxis: importing docker image: ubuntu:22.04
# pyxis: imported docker image: ubuntu:22.04
# Pyxis container works!
```

✅ Full stack verified.

---

## Step 6 — Set Up Scratch Space

```bash
export MLPERF_SCRATCH_PATH=/home/ark/scratch/space
mkdir -p $MLPERF_SCRATCH_PATH/data $MLPERF_SCRATCH_PATH/models $MLPERF_SCRATCH_PATH/preprocessed_data
echo 'export MLPERF_SCRATCH_PATH=/home/ark/scratch/space' >> ~/.bashrc
source ~/.bashrc
```

> **Note:** Official README recommends 10TB for all benchmarks. For llama2-70b only (~390GB total) the current 1.5TB free is sufficient. Add a second U.2 drive via LVM before running llama3.1-405b (~1.4TB).

> **Note:** Enroot does not inherit host env vars. Always pass `MLPERF_SCRATCH_PATH` explicitly on the make command line when building the container.

---

## Step 7 — Link Build Directories and Clone 3rdparty

```bash
cd ~/inference_results_v6.0/closed/NVIDIA
make link_dirs
ls -al build/
# build/data -> /home/ark/scratch/space/data
# build/models -> /home/ark/scratch/space/models
# build/preprocessed_data -> /home/ark/scratch/space/preprocessed_data

# Clone 3rdparty deps
mkdir -p 3rdparty
git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git 3rdparty/trtllm
git clone --depth 1 https://github.com/mlcommons/inference.git 3rdparty/mlc-inference
```

> **Warning:** `make clean` deletes `3rdparty/trtllm` and `3rdparty/mlc-inference`. Never run `make clean` after cloning — re-clone if needed.

---

## Step 8 — Build Container and Enter MLPerf Environment

`sqsh` format is required for scaleout/Pyxis runs (not Docker). The same `make prebuild` command both builds the container (first run) and launches an interactive shell (subsequent runs) — it automatically skips already-built layers.

```bash
make prebuild BENCHMARK=llama2-70b ENV=release PREBUILD_TYPE=sqsh \
    MLPERF_SCRATCH_PATH=/home/ark/scratch/space
# First run: builds sqsh images (~20-60 min, downloads ~20-30GB)
# Subsequent runs: skips build, drops straight into: root@ark:/work#
```

Inside the container, set scratch path and verify:

```bash
export MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage
make link_dirs
ls -al build/
nvidia-smi  # All 8x B300 should be visible
```

> **Note:** First run downloads `nvcr.io/nvidia/pytorch:26.02-py3` (~20-30GB). Monitor progress with `ifstat 2 5` on `eno2` in another terminal.

> **Note:** `Makefile.pyxis` has a hardcoded default `MLPERF_SCRATCH_PATH=/lustre/share/...` (NVIDIA internal Lustre path). Always override on the command line.

> **Note:** `$MLPERF_SCRATCH_PATH` appears empty inside the container even though scratch is correctly mounted at `/home/mlperf_inference_storage`. Always re-export manually after entering.

### Abnormal Exit / Stale Job Recovery

The container session runs as a **Slurm job**. If you close the terminal, reboot, or get disconnected without typing `exit`, the Slurm job keeps running and holds the node in `mix` state — blocking future container launches with `srun: job queued and waiting for resources`.

**Symptoms:**
```bash
sinfo   # node shows "mix" instead of "idle"
squeue  # shows a stale job running for days
```

**Remedy:**
```bash
# Find and cancel the stale job
squeue
scancel <JOBID>

# Verify node is idle
sinfo
# PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
# b300*        up   infinite      1   idle ark
```

**After reboot**, also re-apply enroot runtime dir (if `tmpfiles.d` didn't trigger):
```bash
sudo mkdir -p /run/enroot
sudo chmod 777 /run/enroot
sudo mkdir -p /run/enroot/user/$(id -u)
sudo chown -R $USER:$USER /run/enroot/user/$(id -u)
```

> **Best practice:** Always type `exit` to leave the container cleanly — this releases the Slurm job and returns the node to `idle` immediately.

---

## Step 9 — Download Models and Dataset

> **Reference:** [Llama2-70b Benchmark README](https://github.com/mlcommons/inference_results_v6.0/blob/main/closed/NVIDIA/code/llama2-70b/tensorrt/README.md)

All downloads run in parallel across multiple terminals to save time. Models download inside the container; datasets download on the host.

### Model Downloads (inside container)

```bash
# Set HF token — required for gated repos
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# Terminal 1 — Base model (~140GB safetensors only)
mkdir -p build/models/Llama2/Llama-2-70b-chat-hf
hf download meta-llama/Llama-2-70b-chat-hf \
    --local-dir build/models/Llama2/Llama-2-70b-chat-hf \
    --ignore-patterns="*.bin"

# Terminal 2 — NVFP4 quantized checkpoint (~39.6GB)
mkdir -p build/models/Llama2/llama2-70b-chat-hf-torch-fp4
hf download centml/llama2-70b-chat-hf-torch-fp4_mlperf-inf-v6.0 \
    --local-dir build/models/Llama2/llama2-70b-chat-hf-torch-fp4
```

> **Note:** `huggingface-cli` is not available inside the container — use `hf` instead.

> **Note:** Must accept HF license agreements on HuggingFace website before downloading, otherwise download returns "Repository not found" even with valid token:
> - https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
> - https://huggingface.co/centml/llama2-70b-chat-hf-torch-fp4_mlperf-inf-v6.0

> **Warning:** HuggingFace repo contains **both** `.safetensors` and `.bin` formats (~140GB each). Without `--ignore-patterns="*.bin"`, both are downloaded, wasting ~140GB. Always use `--ignore-patterns="*.bin"` — safetensors is sufficient and preferred. If already downloaded both, delete `.bin` files to reclaim space:
> ```bash
> rm build/models/Llama2/Llama-2-70b-chat-hf/pytorch_model-*.bin
> rm build/models/Llama2/Llama-2-70b-chat-hf/pytorch_model.bin.index.json
> ```

> **Note:** v6.0 introduced a pre-quantized NVFP4 checkpoint on HuggingFace as a convenience. Previous versions required quantizing locally via `code/llmlib/hf_quantize.py`. We use the pre-quantized path.

### Dataset Downloads (on host, outside container)

The original HuggingFace dataset URLs (`open-orca/1million-gpt-4`) return **404** — the dataset has moved to `inference.mlcommons-storage.org`. Use MLCFlow which handles download + checksum verification automatically:

```bash
# Install mlcflow on host
pip install mlcflow --break-system-packages

# mlcflow installs to ~/.local/bin which is not on PATH by default
export PATH=$HOME/.local/bin:$PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
```

**Important:** First `mlcr` run tries to auto-clone `mlcommons/mlperf-automations` (90k objects) which **times out on slow networks**. Clone manually with `--depth 1` first:

```bash
# Manual shallow clone workaround (mlc pull repo ignores --depth internally)
git clone --depth 1 --branch dev \
    https://github.com/mlcommons/mlperf-automations.git \
    ~/MLC/repos/mlcommons@mlperf-automations
```

Then run the dataset download (validation and calibration can run in separate terminals):

```bash
# Terminal 3 — Validation dataset
mlcr get,dataset,preprocessed,openorca,_validation,_r2-downloader,_mlc \
    --outdirname=/home/ark/scratch/space/data/llama2-70b -j

# Terminal 4 — Calibration dataset
mlcr get,dataset,preprocessed,openorca,_calibration,_r2-downloader,_mlc \
    --outdirname=/home/ark/scratch/space/data/llama2-70b -j
```

MLCFlow downloads from `inference.mlcommons-storage.org` via the r2-downloader script. Files downloaded and MD5-verified:

```
open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz   ✅ (30MB)
open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz ✅ (1.3MB)
reference_impl_gpu_bs32_fp32_output.pkl.gz             ✅ (50MB)
mlperf_log_accuracy.json                               ✅ (316MB)
```

Both `.pkl.gz` files are auto-unzipped to `.pkl` by MLCFlow.

> **Note:** MLCFlow installs `transformers`, `pyarrow`, `fastparquet`, `pandas` as dependencies — it needs the Llama2 tokenizer for preprocessing pipeline even for the preprocessed download path.

> **Note:** `mlcr` runs on the **host** (outside container), not inside the container.

### Monitor All Downloads

```bash
watch -n60 'du -sh /home/ark/scratch/space/models/Llama2/* /home/ark/scratch/space/data/llama2-70b/'
```

### Final Download Summary

| Item | Size | Location |
|---|---|---|
| Base model (safetensors) | ~140GB | `build/models/Llama2/Llama-2-70b-chat-hf/` |
| NVFP4 checkpoint | ~39.6GB | `build/models/Llama2/llama2-70b-chat-hf-torch-fp4/` |
| Dataset (validation + calibration) | ~400MB | `build/data/llama2-70b/llama-2-70b-open-orca-dataset.uri/` |
| Disk free after cleanup | ~1.2TB | ✅ Healthy |

---

## Step 10 — Preprocess Dataset (inside container)

> **Reference:** [Llama2-70b Benchmark README](https://github.com/mlcommons/inference_results_v6.0/blob/main/closed/NVIDIA/code/llama2-70b/tensorrt/README.md) — "Download and Prepare Data" section but MLPerf still needs them converted to padded numpy format via `preprocess_data.py`:

```bash
# Inside container
export MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage

# Copy pkl files to expected location
mkdir -p build/data/llama2-70b
cp /home/mlperf_inference_storage/data/llama2-70b/llama-2-70b-open-orca-dataset.uri/open_orca_gpt4_tokenized_llama.sampled_24576.pkl \
    build/data/llama2-70b/
cp /home/mlperf_inference_storage/data/llama2-70b/llama-2-70b-open-orca-dataset.uri/open_orca_gpt4_tokenized_llama.calibration_1000.pkl \
    build/data/llama2-70b/

# Run preprocessing (~10-30 min)
python3 code/llama2-70b/tensorrt/preprocess_data.py \
    --data_dir build/data/ \
    --preprocessed_data_dir build/preprocessed_data
```

Expected output files after preprocessing:

```
build/preprocessed_data/llama2-70b/
  ├── input_lens.npy
  ├── input_ids_padded.npy
  └── mlperf_llama2_openorca_calibration_1k/data.parquet
```

---

## Step 11 — Build TensorRT-LLM from Source (tensorrt_llm missing from sqsh)

### Why this is needed

The `ENV=release` sqsh build installs `tensorrt_llm` via `make build_trt_llm && make install_trt_llm` inside the container. However this step is blocked by `Makefile.docker` line 83 which checks `MLPERF_SCRATCH_PATH` even during the trtllm-release build — an NVIDIA internal assumption that scratch always exists at their Lustre path. The build exits with error code 2 but still exports the sqsh — meaning the sqsh is saved **before** tensorrt_llm is installed.

### Missing dependencies after entering container

After entering the container, several packages are missing and must be installed manually each session (container is stateless — overlay is discarded on exit):

```bash
# nvmitten — not on PyPI, install from GitHub
pip install git+https://github.com/NVIDIA/mitten@4b6d18d76a530cfad4c70302d93d9afd68e7d99d \
    --break-system-packages -q

# onnx_graphsurgeon
pip install onnx_graphsurgeon --break-system-packages -q

# transformers
pip install transformers --break-system-packages -q

# mlperf_loadgen — build from cloned mlc-inference
cd /work/3rdparty/mlc-inference/loadgen
pip install . --break-system-packages -q
cd /work
```

To avoid repeating this every session, create a `setup_deps.sh` script **on the host** (persists across container sessions since `/work` is mounted):

```bash
# Create on host at ~/inference_results_v6.0/closed/NVIDIA/setup_deps.sh
cat > ~/inference_results_v6.0/closed/NVIDIA/setup_deps.sh << 'EOF'
#!/bin/bash
echo "Installing missing dependencies..."
pip install git+https://github.com/NVIDIA/mitten@4b6d18d76a530cfad4c70302d93d9afd68e7d99d \
    --break-system-packages -q
pip install onnx_graphsurgeon --break-system-packages -q
pip install transformers --break-system-packages -q
cd /work/3rdparty/mlc-inference/loadgen
pip install . --break-system-packages -q
cd /work
echo "Done!"
EOF
chmod +x ~/inference_results_v6.0/closed/NVIDIA/setup_deps.sh
```

Then every time you enter the container, just run:

```bash
# Inside container
bash /work/setup_deps.sh
```

> **Note:** `setup_deps.sh` lives in `/work` (host-mounted), so it persists across container sessions even though the container overlay is discarded on exit. `tensorrt_llm` is NOT in this script — it must be baked into the sqsh image (see below).

> **Note:** Also missing `3rdparty/mitten` submodule — clone it if `install_mitten.sh` fails:
> ```bash
> # On host, outside container
> git clone https://github.com/NVIDIA/mitten.git \
>     ~/inference_results_v6.0/closed/NVIDIA/3rdparty/mitten
> ```

### Building tensorrt_llm from source

The PyPI version of `tensorrt_llm` conflicts with Slurm PMIx and crashes on import. Must build from source:

```bash
# On host — pull Git LFS objects first (required for cutlass kernels)
# cloning with --depth 1 does NOT pull LFS objects
cd ~/inference_results_v6.0/closed/NVIDIA/3rdparty/trtllm
sudo apt install git-lfs -y
git lfs install
git lfs pull
cd ~/inference_results_v6.0/closed/NVIDIA
```

Then build inside the trtllm-dev sqsh (bypassing Makefile.docker path check):

```bash
srun --partition=b300 \
    --container-image "$(pwd)/build/sqsh_images/trtllm-dev-5277e6252eaf91f65ae60bc62b233e5284d7d33f.sqsh" \
    --container-save "$(pwd)/build/sqsh_images/trtllm-release-5277e6252eaf91f65ae60bc62b233e5284d7d33f.sqsh" \
    --container-mounts "$(pwd):/work" \
    --container-workdir /work/3rdparty/trtllm \
    --container-remap-root \
    bash -c 'python scripts/build_wheel.py --use_ccache --benchmarks -a="90-real;100-real;103-real" --no-venv --clean && pip install build/tensorrt*.whl'
```

> **Note:** Build architectures: `90-real` (H100), `100-real` (B200), `103-real` (B300). Takes 60-90 minutes.

> **Note:** Use absolute path with `$(pwd)/` prefix for sqsh files — Pyxis distinguishes local sqsh from registry images by leading `/`. Relative paths are treated as Docker registry URLs and fail with 401.

> **Note:** `git lfs pull` is required — `--depth 1` clone does not pull LFS binary objects. Build fails with `internal_cutlass_kernels library is truncated` without it.

After build completes, rebuild the MLPerf sqsh on top:

```bash
rm build/sqsh_images/mlperf-inference-ark-x86_64-release-llama2-70b.sqsh

make prebuild BENCHMARK=llama2-70b ENV=release PREBUILD_TYPE=sqsh \
    MLPERF_SCRATCH_PATH=/home/ark/scratch/space
```

Verify inside container:

```bash
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
python3 -c "import mlperf_loadgen; print('loadgen OK')"
python3 -c "import nvmitten; print('nvmitten OK')"
```
- **`tensorrt_llm` PyPI version conflicts with Slurm PMIx** — crashes on import with `MPI_Init_thread` error; must build from source instead
- **`make build_trt_llm` blocked by `MLPERF_SCRATCH_PATH` check** — bypass by running `srun` directly with absolute sqsh paths instead of going through `make prebuild`
- **`git clone --depth 1` does not pull Git LFS objects** — run `git lfs install && git lfs pull` in `3rdparty/trtllm` before building, otherwise build fails with `internal_cutlass_kernels library is truncated`
- **Pyxis treats relative sqsh paths as Docker registry URLs** — always use absolute paths (`$(pwd)/...`) for local sqsh files in `srun --container-image`
- **Container is stateless** — all pip installs (nvmitten, onnx_graphsurgeon, loadgen) are lost on exit; create `setup_deps.sh` in `/work` to reinstall quickly
- **`3rdparty/mitten` is a required submodule** not cloned by default — clone manually if `install_mitten.sh` fails during sqsh build

---


---

## Run Log

### Run #1 — YYYY-MM-DD

| Field | Value |
|---|---|
| Model | llama2-70b |
| Scenario | Offline |
| Precision | NVFP4 |
| GPUs | 8x B300 |
| Status | ✅ Pass / ❌ Fail |

**Command:**
```bash
salloc --nodes=1 --partition=b300
./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system B300-SXM-270GBx1 \
    --gpus-per-node 8 \
    --dp-multiplicity 8 \
    --run-args "--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint"
```

**Results:**

| Metric | Value |
|---|---|
| Throughput (tok/s) | |
| Latency P99 (ms) | |
| Accuracy | |

**Notes:**
>

---

### Run #2 — YYYY-MM-DD

| Field | Value |
|---|---|
| Model | llama2-70b |
| Scenario | Server |
| Precision | NVFP4 |
| GPUs | 8x B300 |
| Status | ✅ Pass / ❌ Fail |

**Command:**
```bash
salloc --nodes=1 --partition=b300
./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system B300-SXM-270GBx1 \
    --gpus-per-node 8 \
    --dp-multiplicity 8 \
    --run-args "--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint --server_target_qps=400"
```

**Results:**

| Metric | Value |
|---|---|
| Throughput (tok/s) | |
| TTFT P99 (ms) | |
| TPS/user P99 | |

**Notes:**
>

---

### Run #3 — YYYY-MM-DD

| Field | Value |
|---|---|
| Model | llama3.1-405b |
| Scenario | Offline |
| Precision | NVFP4 |
| GPUs | 8x B300 |
| Status | ✅ Pass / ❌ Fail |

**Command:**
```bash
salloc --nodes=1 --partition=b300
./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system B300-SXM-270GBx2 \
    --gpus-per-node 8 \
    --dp-multiplicity 4 \
    --run-args "--benchmarks=llama3.1-405b --scenarios=Offline --core_type=trtllm_endpoint"
```

**Results:**

| Metric | Value |
|---|---|
| Throughput (tok/s) | |
| TTFT P99 (ms) | |
| TPS/user P99 | |

**Notes:**
>

---

---

## Summary

| Model | Scenario | Throughput (tok/s) | Pass? |
|---|---|---|---|
| llama2-70b | Offline | | |
| llama2-70b | Server | | |
| llama3.1-405b | Offline | | |

---

## Issues & Lessons Learned

- `SlurmUser=slurm` must be explicitly declared in `slurm.conf` to avoid security violation errors for uid 64030
- Pyxis has no prebuilt `.deb` — always build from source matching your Slurm version
- Download `.deb` files to `/tmp/` to avoid `_apt` permission warnings
- Enroot `tput`/`$TERM` warnings on version check are cosmetic and harmless
- PMIx warnings in slurmd are harmless for MLPerf (uses NCCL, not MPI)
- `deepseek-r1` and `llama3.1-8b` have no scaleout `REPRODUCE.md` commands for B300x8 — use Docker path instead
- **Always `exit` cleanly from container** — disconnecting without exit leaves a stale Slurm job holding the node in `mix` state; fix with `scancel <JOBID>`
- **After reboot, `/run/enroot` is wiped** — manually recreate if `tmpfiles.d` doesn't trigger, otherwise `make prebuild` fails with `Permission denied`
- **Enroot does not inherit host env vars** unlike Docker — always re-export `MLPERF_SCRATCH_PATH` inside container
- **Ubuntu 24.04 AppArmor** restricts unprivileged user namespaces by default — must disable with `sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` or enroot fails with `Read-only file system`
- **System-wide `/etc/enroot/enroot.conf`** must have `ENROOT_ROOTFS_WRITABLE=yes` — user-level config alone is not enough because slurmd runs as a different user
- **`make clean` deletes 3rdparty clones** — never run it after cloning trtllm and mlc-inference
- **`Makefile.pyxis` hardcodes NVIDIA's internal scratch path** (`/lustre/share/...`) — always override with `MLPERF_SCRATCH_PATH=/home/ark/scratch/space` on the make command line
- **`make prebuild` serves dual purpose** — builds sqsh on first run, launches interactive shell on subsequent runs (same command, no separate "enter" step needed)
- **HuggingFace downloads both `.safetensors` and `.bin`** — use `--ignore-patterns="*.bin"` to avoid doubling storage (saves ~140GB for llama2-70b). If already downloaded both, delete `.bin` files manually
- **HuggingFace dataset URLs are stale** — `open-orca/1million-gpt-4` returns 404; dataset moved to `inference.mlcommons-storage.org`; use `mlcr get,dataset,preprocessed,openorca,...` instead
- **`mlc pull repo` ignores `--depth` flag** — always clone `mlperf-automations` manually with `git clone --depth 1` to avoid timeout on slow networks
- **MLCFlow installs `transformers`** because it runs the full pipeline (download + tokenize + preprocess), not just a wget — replaces the manual `preprocess_data.py` step
- **`mlcr` runs on host, not inside container** — dataset preprocessing happens outside the MLPerf container

---
