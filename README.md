# Env2Scaffold: ALFWorld Environment Tuning Experiments

Automated environment augmentation + RL training for ALFWorld agents. Based on [Benchmark2Scaffold](https://github.com/Ch1nyzzz/env2scaffold/tree/main/env2scaffold) (auto-discovers environment feedback rules) and [Environment Tuning](https://arxiv.org/abs/2510.10197) (trains agents with augmented environments).

Training framework: [verl-agent (GiGPO)](https://github.com/langfengQ/verl-agent)

## Three Experiments

| Experiment | Environment | Reward | Script |
|---|---|---|---|
| **Vanilla GRPO** | Original ALFWorld | Sparse (`10 * won`) | `run_alfworld_vanilla.sh` |
| **Obs-Aug GRPO** | Augmented feedback text | Sparse (`10 * won`) | `run_alfworld_envtuning.sh` |
| **Full EnvTuning GRPO** | Augmented feedback text | Sparse + progress reward | `run_alfworld_full_envtuning.sh` |

## Quick Start with Docker

### Prerequisites

- NVIDIA GPU (8x A100 80GB recommended)
- Docker with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- ~20GB disk for model

### Step 1: Pull Docker Image

```bash
docker pull yuhan778/alfworld-envtuning:latest
```

### Step 2: Download Model

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-8B --local-dir /path/to/models/Qwen3-8B
```

### Step 3: Run Experiments

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

### Monitoring

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
│   │   └── run_alfworld_full_envtuning.sh
│   └── agent_system/environments/env_package/alfworld/
│       └── envs.py            # AugmentedAlfworldEnvs injection
├── AWorld-RL/EnvTuning/       # Original EnvTuning codebase (reference)
├── Dockerfile
└── README.md
```
