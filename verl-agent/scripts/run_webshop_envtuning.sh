#!/bin/bash
# WebShop Obs-Aug GRPO: generated env2scaffold observation feedback, sparse success reward.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
cd "$REPO_ROOT"

ENGINE="${ENGINE:-vllm}"
if [[ $# -gt 0 && "$1" != *=* && "$1" != +* && "$1" != -* ]]; then
  ENGINE="$1"
  shift
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
elif [[ -f "$WORKSPACE_ROOT/venv310/bin/activate" ]]; then
  source "$WORKSPACE_ROOT/venv310/bin/activate"
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export ENV2SCAFFOLD_WEBSHOP_AUG_PATH="${ENV2SCAFFOLD_WEBSHOP_AUG_PATH:-$WORKSPACE_ROOT/env2scaffold/generated/webshop_codex/augmentation/augmented_env.py}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_CONFIGURE_LOGGING="${VLLM_CONFIGURE_LOGGING:-0}"
export RAY_DISABLE_TELEMETRY="${RAY_DISABLE_TELEMETRY:-1}"
export OTEL_SDK_DISABLED="${OTEL_SDK_DISABLED:-true}"

MODEL_PATH="${MODEL_PATH:-/data/home/yuhan/model_zoo/Qwen3-8B}"
if [[ ! -e "$MODEL_PATH" ]]; then
  MODEL_PATH="${FALLBACK_MODEL_PATH:-Qwen/Qwen3-8B}"
fi

WEBSHOP_DIR="$REPO_ROOT/agent_system/environments/env_package/webshop/webshop"
if [[ ! -f "$WEBSHOP_DIR/data/items_shuffle_1000.json" || ! -d "$WEBSHOP_DIR/search_engine/indexes" ]]; then
  echo "Missing WebShop small data/index. Run from workspace root:" >&2
  echo "  DATA_SIZE=small bash env2scaffold/generated/webshop_codex/scripts/setup_webshop_data.sh" >&2
  exit 2
fi
if [[ ! -f "$ENV2SCAFFOLD_WEBSHOP_AUG_PATH" ]]; then
  echo "Missing WebShop obs-aug wrapper: $ENV2SCAFFOLD_WEBSHOP_AUG_PATH" >&2
  exit 2
fi

TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG_DIR="${LOG_DIR:-logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKSPACE_ROOT/outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen3-8B-webshop-obs-aug-grpo-${TIMESTAMP}}"
ROLLOUT_DIR="$OUTPUT_ROOT/rollout/webshop-obs-aug-grpo/${EXPERIMENT_NAME}"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
N_GPUS="${N_GPUS:-8}"
TP="${TP:-2}"
MAX_STEPS="${MAX_STEPS:-15}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-150}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$TRAIN_DATA_SIZE}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$VAL_DATA_SIZE}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
WEBSHOP_USE_SMALL="${WEBSHOP_USE_SMALL:-true}"
WEBSHOP_HUMAN_GOALS="${WEBSHOP_HUMAN_GOALS:-false}"

python3 -m examples.data_preprocess.prepare \
  --mode text \
  --train_data_size "$TRAIN_DATA_SIZE" \
  --val_data_size "$VAL_DATA_SIZE"

LOG_FILE="$LOG_DIR/webshop_obs_aug_${TIMESTAMP}.log"
ERR_FILE="$LOG_DIR/webshop_obs_aug_${TIMESTAMP}.err"
PROGRESS_FILE="$LOG_DIR/webshop_obs_aug_progress.log"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$HOME/data/verl-agent/text/train.parquet" \
  data.val_files="$HOME/data/verl-agent/text/test.parquet" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
  actor_rollout_ref.rollout.name="$ENGINE" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  env.env_name=Webshop \
  +env.use_augmented_env=True \
  +env.use_progress_reward=False \
  env.seed=0 \
  env.max_steps="$MAX_STEPS" \
  env.history_length=2 \
  env.rollout.n="$GROUP_SIZE" \
  env.resources_per_worker.num_cpus="$NUM_CPUS_PER_ENV_WORKER" \
  env.webshop.use_small="$WEBSHOP_USE_SMALL" \
  env.webshop.human_goals="$WEBSHOP_HUMAN_GOALS" \
  trainer.critic_warmup=0 \
  trainer.logger="['console','wandb']" \
  trainer.project_name=webshop-grpo \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.val_before_train=True \
  trainer.rollout_data_dir="$ROLLOUT_DIR" \
  trainer.default_local_dir="${CKPT_DIR:-$OUTPUT_ROOT/checkpoints}/webshop-obs-aug-grpo" \
  "$@" >"$LOG_FILE" 2>"$ERR_FILE" &

TRAIN_PID=$!
{
  echo "PID: $TRAIN_PID"
  echo "Experiment: $EXPERIMENT_NAME"
  echo "Started: $(date)"
  echo "Log: $LOG_FILE"
  echo "Err: $ERR_FILE"
  echo "Rollout Dir: $ROLLOUT_DIR"
  echo "Obs-Aug Path: $ENV2SCAFFOLD_WEBSHOP_AUG_PATH"
} > "$PROGRESS_FILE"

set +e
wait "$TRAIN_PID"
EXIT_CODE=$?
set -e
echo "Finished: $(date), exit code: $EXIT_CODE" >> "$PROGRESS_FILE"
exit "$EXIT_CODE"
