# Env2Scaffold: Environment Tuning Experiments

Automated environment augmentation + RL training for ALFWorld and WebShop agents. Based on [Benchmark2Scaffold](https://github.com/Ch1nyzzz/env2scaffold/tree/main/env2scaffold) (auto-discovers environment feedback rules) and [Environment Tuning](https://arxiv.org/abs/2510.10197) (trains agents with augmented environments).

Training framework: [verl-agent (GiGPO)](https://github.com/langfengQ/verl-agent)

## ALFWorld Experiments

| Experiment | Environment | Reward | Script |
|---|---|---|---|
| **Vanilla GRPO** | Original ALFWorld | Sparse (`10 * won`) | `run_alfworld_vanilla.sh` |
| **Obs-Aug GRPO** | Augmented feedback text | Sparse (`10 * won`) | `run_alfworld_envtuning.sh` |
| **Full EnvTuning GRPO** | Augmented feedback text | Sparse + progress reward | `run_alfworld_full_envtuning.sh` |

## WebShop Experiments

| Experiment | Environment | Reward | Script |
|---|---|---|---|
| **Vanilla GRPO** | Original WebShop | Sparse (`10 * success`) | `run_webshop_vanilla.sh` |
| **Obs-Aug GRPO** | Augmented feedback text | Sparse (`10 * success`) | `run_webshop_envtuning.sh` |
| **Full EnvTuning GRPO** | Augmented feedback text | Sparse + progress reward | `run_webshop_full_envtuning.sh` |

## Quick Start with Docker

### ALFWorld Image

#### Prerequisites

- NVIDIA GPU (8x A100 80GB recommended)
- Docker with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- ~20GB disk for model

#### Step 1: Pull Docker Image

```bash
docker pull yuhan778/alfworld-envtuning:latest
```

#### Step 2: Download Model

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-8B --local-dir /path/to/models/Qwen3-8B
```

#### Step 3: Run Experiments

```bash
# Vanilla GRPO (baseline)
docker run --gpus all --ipc=host --shm-size=64g \
    -e MODEL_PATH=/models/Qwen3-8B \
    -e CKPT_DIR=/workspace/outputs/checkpoints \
    -v /path/to/models/Qwen3-8B:/models/Qwen3-8B \
    -v /path/to/outputs:/workspace/outputs \
    yuhan778/alfworld-envtuning:latest \
    bash scripts/run_alfworld_vanilla.sh

# Obs-Aug GRPO (augmented observation text only)
docker run --gpus all --ipc=host --shm-size=64g \
    -e MODEL_PATH=/models/Qwen3-8B \
    -e CKPT_DIR=/workspace/outputs/checkpoints \
    -v /path/to/models/Qwen3-8B:/models/Qwen3-8B \
    -v /path/to/outputs:/workspace/outputs \
    yuhan778/alfworld-envtuning:latest \
    bash scripts/run_alfworld_envtuning.sh

# Full EnvTuning GRPO (augmented obs + progress reward)
docker run --gpus all --ipc=host --shm-size=64g \
    -e MODEL_PATH=/models/Qwen3-8B \
    -e CKPT_DIR=/workspace/outputs/checkpoints \
    -v /path/to/models/Qwen3-8B:/models/Qwen3-8B \
    -v /path/to/outputs:/workspace/outputs \
    yuhan778/alfworld-envtuning:latest \
    bash scripts/run_alfworld_full_envtuning.sh
```

#### Monitoring

Training logs to [wandb](https://wandb.ai). Set your own key:

```bash
docker run --gpus all --ipc=host --shm-size=64g \
    -e MODEL_PATH=/models/Qwen3-8B \
    -e CKPT_DIR=/workspace/outputs/checkpoints \
    -e WANDB_API_KEY=your_wandb_key \
    -v /path/to/models/Qwen3-8B:/models/Qwen3-8B \
    -v /path/to/outputs:/workspace/outputs \
    yuhan778/alfworld-envtuning:latest \
    bash scripts/run_alfworld_vanilla.sh
```

Checkpoints saved every 20 steps to the mounted outputs directory.

### WebShop Image

WebShop needs its product JSON files plus a Lucene/Pyserini search index. The WebShop training image in this repo packages the small WebShop data/index by default, so the runtime container does not need a separate `setup.sh` step. The model is still mounted at runtime instead of baked into the image.

Pull the published image:

```bash
docker pull yuhan778/webshop-envtuning:latest
```

Or build the image locally:

```bash
bash build_and_run_webshop.sh build
```

Optionally export it for another machine:

```bash
bash build_and_run_webshop.sh export
scp webshop-envtuning.tar.gz user@other-machine:/path/

# On the other machine:
bash build_and_run_webshop.sh load
```

Download the model on the host:

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-8B --local-dir /path/to/models/Qwen3-8B
```

Run WebShop experiments:

```bash
# Vanilla GRPO
MODEL_HOST_PATH=/path/to/models/Qwen3-8B \
OUTPUT_HOST_PATH=/path/to/outputs \
bash build_and_run_webshop.sh vanilla

# Obs-Aug GRPO
MODEL_HOST_PATH=/path/to/models/Qwen3-8B \
OUTPUT_HOST_PATH=/path/to/outputs \
bash build_and_run_webshop.sh obs-aug

# Full EnvTuning GRPO
MODEL_HOST_PATH=/path/to/models/Qwen3-8B \
OUTPUT_HOST_PATH=/path/to/outputs \
bash build_and_run_webshop.sh full-envtuning
```

Useful overrides:

```bash
# Match smaller GPU nodes.
N_GPUS=2 TP=2 MODEL_HOST_PATH=/path/to/models/Qwen3-8B OUTPUT_HOST_PATH=/path/to/outputs \
bash build_and_run_webshop.sh vanilla

# Forward wandb auth into the container.
WANDB_API_KEY=your_wandb_key MODEL_HOST_PATH=/path/to/models/Qwen3-8B OUTPUT_HOST_PATH=/path/to/outputs \
bash build_and_run_webshop.sh full-envtuning
```

Direct Docker commands are equivalent:

```bash
docker run --gpus all --ipc=host --shm-size=64g \
    --ulimit nproc=65536:65536 \
    --ulimit nofile=1048576:1048576 \
    -e MODEL_PATH=/models/Qwen3-8B \
    -e CKPT_DIR=/workspace/outputs/checkpoints \
    -e OUTPUT_ROOT=/workspace/outputs \
    -e LOG_DIR=/workspace/outputs/logs \
    -e N_GPUS=8 \
    -e TP=2 \
    -e WEBSHOP_USE_SMALL=true \
    -v /path/to/models/Qwen3-8B:/models/Qwen3-8B:ro \
    -v /path/to/outputs:/workspace/outputs \
    webshop-envtuning:latest \
    bash scripts/run_webshop_full_envtuning.sh
```

The packaged image targets the small WebShop data path used by the current training scripts (`WEBSHOP_USE_SMALL=true`). If you want to train on the full WebShop product files later, regenerate both the full JSON files and matching Lucene indexes first.

## Setup from Source (without Docker)

### Requirements

- Python 3.10
- CUDA 12.4+
- 8x NVIDIA A100 80GB (or equivalent)

### Installation

```bash
# Clone
git clone https://github.com/Ch1nyzzz/env2scaffold.git
cd env2scaffold

# Create Python 3.10 venv
python3.10 -m venv venv310
source venv310/bin/activate

# Install core deps (matching GiGPO paper)
pip install "vllm==0.8.3" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata
pip install "ray[default]" codetiming hydra-core pylatexenc wandb pybind11 datasets accelerate peft
pip install alfworld textworld "transformers>=4.51,<4.53" "setuptools<81" gymnasium

# Install flash-attn (prebuilt wheel)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install verl-agent
cd verl-agent && pip install -e . && cd ..

# Download ALFWorld game data
alfworld-download

# Prepare WebShop small data/index if running WebShop without Docker.
# WebShop setup.sh expects conda for faiss/openjdk.
cd verl-agent/agent_system/environments/env_package/webshop/webshop
./setup.sh -d small
cd ../../../../../..

# Download model
huggingface-cli download Qwen/Qwen3-8B --local-dir /path/to/models/Qwen3-8B
```

### Run Training

```bash
cd verl-agent
export MODEL_PATH=/path/to/models/Qwen3-8B
bash scripts/run_alfworld_vanilla.sh          # Vanilla
bash scripts/run_alfworld_envtuning.sh        # Obs-Aug
bash scripts/run_alfworld_full_envtuning.sh   # Full EnvTuning
bash scripts/run_webshop_vanilla.sh           # WebShop Vanilla
bash scripts/run_webshop_envtuning.sh         # WebShop Obs-Aug
bash scripts/run_webshop_full_envtuning.sh    # WebShop Full EnvTuning
```

## Training Hyperparameters

Aligned with [GiGPO paper](https://arxiv.org/abs/2505.10978):

| Parameter | Value |
|---|---|
| Base model | Qwen3-8B |
| train_batch_size | 16 |
| group_size (rollout.n) | 8 |
| total_epochs | 150 |
| lr | 1e-6 |
| kl_loss_coef | 0.01 |
| max_prompt_length | 2048 |
| max_response_length | 512 |
| tensor_parallel_size | 2 |
| gpu_memory_utilization | 0.6 |
| micro_batch_size_per_gpu | 8 |
| param_offload | True |
| optimizer_offload | True |
| max_env_steps | 50 |
| test_freq | 5 |
| save_freq | 20 |

## Project Structure

```
env-aug/
├── env2scaffold/              # Benchmark2Scaffold: auto environment augmentation
│   ├── augmentation/
│   │   └── augmented_env.py   # AugmentedAlfWorldEnv wrapper
│   ├── probing/               # Environment probing agent
│   ├── benchmark_spec/        # Stage 1: benchmark reader output
│   ├── audit/                 # Stage 3: feedback audit + candidates
│   ├── oracle_test/           # Pipeline B: oracle & test plan
│   ├── verification/          # 3-layer verification runner
│   └── pipeline.py            # Multi-agent augmentation pipeline
├── verl-agent/                # Training framework (modified verl-agent)
│   ├── scripts/
│   │   ├── run_alfworld_vanilla.sh
│   │   ├── run_alfworld_envtuning.sh
│   │   ├── run_alfworld_full_envtuning.sh
│   │   ├── run_webshop_vanilla.sh
│   │   ├── run_webshop_envtuning.sh
│   │   └── run_webshop_full_envtuning.sh
│   └── agent_system/environments/env_package/
│       ├── alfworld/envs.py   # AugmentedAlfworldEnvs injection
│       └── webshop/envs.py    # WebShop obs-aug/progress reward injection
├── AWorld-RL/EnvTuning/       # Original EnvTuning codebase (reference)
├── Dockerfile
├── Dockerfile.webshop
├── build_and_run_webshop.sh
└── README.md
```
