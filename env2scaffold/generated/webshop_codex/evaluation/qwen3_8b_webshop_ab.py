#!/usr/bin/env python3
"""
Evaluate Qwen3-8B on native WebShop versus env2scaffold observation wrapping.

This script intentionally lives under generated/webshop_codex. It does not
modify verl-agent or WebShop. It creates a native WebAgentTextEnv and optionally
wraps it with augmentation/augmented_env.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


GENERATED_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GENERATED_ROOT.parents[2]
DEFAULT_WEBSHOP_ROOT = (
    REPO_ROOT
    / "verl-agent"
    / "agent_system"
    / "environments"
    / "env_package"
    / "webshop"
    / "webshop"
)
DEFAULT_MODEL = "/data/home/yuhan/model_zoo/Qwen3-8B"


PROMPT_TEMPLATE = """You are an expert autonomous agent operating in the WebShop ecommerce environment.
Your task is to: {task_description}

Recent history:
{history}

Current observation:
{observation}

Admissible actions:
[
{available_actions}
]

Choose exactly one admissible action. First reason briefly inside <think>...</think>, then output the action inside <action>...</action>.
Valid actions are search[query] or click[label]."""


@dataclass
class GenerationConfig:
    backend: str
    model_path: str
    temperature: float
    top_p: float
    max_tokens: int
    tensor_parallel_size: int
    openai_base_url: str
    openai_model: str


class Generator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        if config.backend == "vllm":
            from vllm import LLM, SamplingParams

            self.sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )
            self.llm = LLM(
                model=config.model_path,
                tensor_parallel_size=config.tensor_parallel_size,
                trust_remote_code=True,
            )
            self.tokenizer = self.llm.get_tokenizer()
        elif config.backend == "transformers":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
        elif config.backend == "openai":
            import requests

            self.requests = requests
        else:
            raise ValueError(f"Unsupported backend: {config.backend}")

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if self.config.backend == "openai":
            response = self.requests.post(
                f"{self.config.openai_base_url.rstrip('/')}/chat/completions",
                json={
                    "model": self.config.openai_model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=300,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if self.config.backend == "vllm":
            outputs = self.llm.generate([rendered], self.sampling_params)
            return outputs[0].outputs[0].text

        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=self.config.temperature > 0,
                temperature=max(self.config.temperature, 1e-5),
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_augmented_wrapper():
    module_path = GENERATED_ROOT / "augmentation" / "augmented_env.py"
    spec = importlib.util.spec_from_file_location("webshop_augmented_env", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.AugmentedWebShopEnv


def build_native_env(args):
    webshop_root = Path(args.webshop_root).resolve()
    if not webshop_root.exists():
        raise FileNotFoundError(f"WebShop root does not exist: {webshop_root}")
    sys.path.insert(0, str(webshop_root))

    import gym
    import web_agent_site.envs  # noqa: F401 - registers gym env

    kwargs: Dict[str, Any] = {
        "observation_mode": "text",
        "human_goals": args.human_goals,
    }
    if args.file_path:
        kwargs["file_path"] = args.file_path
    if args.attr_path:
        kwargs["attr_path"] = args.attr_path
    if args.num_products is not None:
        kwargs["num_products"] = args.num_products
    env = gym.make("WebAgentTextEnv-v0", **kwargs)
    return env


def available_actions_text(actions: Dict[str, Any]) -> str:
    items: List[str] = []
    if actions.get("has_search_bar"):
        items.append("'search[<your query>]'")
    for clickable in actions.get("clickables", []):
        items.append(f"'click[{clickable}]'")
    return ",\n".join(items)


def extract_task(observation: str) -> str:
    parts = observation.split(" [SEP] ")
    for idx, part in enumerate(parts):
        if part.strip().lower() == "instruction:" and idx + 1 < len(parts):
            return parts[idx + 1].strip()
    match = re.search(r"instruction:\s*(.*?)(?:\[sep\]|$)", observation, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else observation[:300]


def extract_action(text: str) -> str:
    match = re.search(r"<action>(.*?)</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    for pattern in (r"(search\[[^\]]+\])", r"(click\[[^\]]+\])"):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return text.strip()[-80:]


def trim(text: str, limit: int = 2500) -> str:
    if len(text) <= limit:
        return text
    return text[:1200] + "\n...\n" + text[-1200:]


def reset_env(env: Any, session_idx: int):
    try:
        return env.reset(session=session_idx)
    except TypeError:
        return env.reset()


def run_episode(env: Any, generator: Generator, session_idx: int, max_steps: int, history_len: int):
    observation, info = reset_env(env, session_idx)
    task = extract_task(observation)
    history: List[str] = []
    trajectory = []
    final_reward = 0.0
    done = False

    for step_idx in range(max_steps):
        available = env.get_available_actions()
        history_text = "\n".join(history[-history_len:]) if history else "None"
        prompt = PROMPT_TEMPLATE.format(
            task_description=task,
            history=history_text,
            observation=trim(observation),
            available_actions=available_actions_text(available),
        )
        raw = generator.generate(prompt)
        action = extract_action(raw)
        next_observation, reward, done, info = env.step(action)

        trajectory.append({
            "step": step_idx + 1,
            "action": action,
            "reward": float(reward),
            "done": bool(done),
            "model_output": raw,
            "feedback": dict(info or {}).get("webshop_feedback"),
        })
        history.append(f"Observation: {trim(observation, 600)}\nAction: {action}")
        observation = next_observation
        final_reward = float(reward)
        if done:
            break

    return {
        "session_idx": session_idx,
        "task": task,
        "done": done,
        "steps": len(trajectory),
        "task_score": final_reward if done else 0.0,
        "success": bool(done and final_reward == 1.0),
        "trajectory": trajectory,
    }


def summarize(episodes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    episodes = list(episodes)
    scores = [float(item["task_score"]) for item in episodes]
    successes = [1.0 if item["success"] else 0.0 for item in episodes]
    steps = [int(item["steps"]) for item in episodes]
    return {
        "episodes": len(episodes),
        "success_rate": statistics.mean(successes) if successes else 0.0,
        "average_task_score": statistics.mean(scores) if scores else 0.0,
        "average_steps": statistics.mean(steps) if steps else 0.0,
    }


def mode_env(native_env: Any, mode: str):
    if mode == "baseline":
        return native_env
    if mode == "augmented":
        return load_augmented_wrapper()(native_env)
    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--webshop-root", default=str(DEFAULT_WEBSHOP_ROOT))
    parser.add_argument("--backend", choices=["vllm", "transformers", "openai"], default="vllm")
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--openai-model", default="qwen3-8b")
    parser.add_argument("--mode", choices=["baseline", "augmented", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-session", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--history-len", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-products", type=int, default=None)
    parser.add_argument("--human-goals", action="store_true")
    parser.add_argument("--file-path", default=None)
    parser.add_argument("--attr-path", default=None)
    parser.add_argument("--output", default=str(GENERATED_ROOT / "evaluation" / "qwen3_8b_webshop_ab_results.json"))
    args = parser.parse_args()

    # Fail fast before loading an 8B model if WebShop dependencies/data are not ready.
    preflight_env = build_native_env(args)
    close = getattr(preflight_env, "close", None)
    if close is not None:
        close()

    generator = Generator(GenerationConfig(
        backend=args.backend,
        model_path=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        openai_base_url=args.openai_base_url,
        openai_model=args.openai_model,
    ))

    modes = ["baseline", "augmented"] if args.mode == "both" else [args.mode]
    all_results: Dict[str, Any] = {
        "model": args.model,
        "backend": args.backend,
        "episodes_requested": args.episodes,
        "max_steps": args.max_steps,
        "modes": {},
    }

    for mode in modes:
        episodes = []
        native = build_native_env(args)
        env = mode_env(native, mode)
        try:
            for offset in range(args.episodes):
                session_idx = args.start_session + offset
                episode = run_episode(env, generator, session_idx, args.max_steps, args.history_len)
                episodes.append(episode)
                print(
                    f"[{mode}] episode={offset + 1}/{args.episodes} "
                    f"session={session_idx} success={episode['success']} "
                    f"score={episode['task_score']:.3f} steps={episode['steps']}"
                )
        finally:
            close = getattr(native, "close", None)
            if close is not None:
                close()
        all_results["modes"][mode] = {
            "summary": summarize(episodes),
            "episodes": episodes,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, indent=2) + "\n")
    print(json.dumps({mode: all_results["modes"][mode]["summary"] for mode in modes}, indent=2))


if __name__ == "__main__":
    main()
