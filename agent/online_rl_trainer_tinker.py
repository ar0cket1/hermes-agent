"""Tinker API-based trainer for Hermes online RL.

Binary mode keeps the original scalar-reward policy-gradient path.
SDPO mode augments that with frozen-teacher top-K distillation from a
feedback-conditioned prompt, approximating the paper's self-distillation
objective within Tinker's public API.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Tinker model IDs that are known to be available.
TINKER_SUPPORTED_MODELS = {
    "moonshotai/Kimi-K2.5",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
}


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize message content to plain strings."""
    normalized: List[Dict[str, str]] = []
    for message in messages or []:
        role = str(message.get("role") or "user")
        content = message.get("content") or ""
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") in {"text", "input_text", "output_text"}:
                        parts.append(str(item.get("text") or ""))
                elif item is not None:
                    parts.append(str(item))
            content = "\n".join(part for part in parts if part)
        normalized.append({"role": role, "content": str(content)})
    return normalized


def _messages_as_text(messages: List[Dict[str, str]]) -> str:
    """Render messages into a simple role-prefixed transcript."""
    lines: List[str] = []
    for message in messages:
        role = str(message.get("role") or "user").upper()
        content = str(message.get("content") or "")
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines).strip()


def _chat_text(tokenizer, messages: List[Dict[str, str]], *, add_generation_prompt: bool) -> str:
    """Apply a chat template when available, falling back to role-prefixed text."""
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    except Exception:
        pass

    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        lines.append(f"{role}: {msg.get('content', '')}")
    if add_generation_prompt:
        lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def _trim_full_sequence(
    token_ids: List[int],
    assistant_start: int,
    *,
    max_length: int,
) -> tuple[List[int], int]:
    """Trim a prompt+response sequence from the left while preserving tail tokens."""
    if len(token_ids) <= max_length:
        return token_ids, assistant_start

    trim = len(token_ids) - max_length
    trimmed = token_ids[trim:]
    new_assistant_start = max(0, assistant_start - trim)
    return trimmed, new_assistant_start


def _build_tinker_examples(
    export_rows: List[Dict[str, Any]],
    tokenizer,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert exported feedback rows into tokenized training examples."""
    max_length = max(512, int(cfg.get("max_sequence_length", 4096)))
    examples: List[Dict[str, Any]] = []
    for row in export_rows:
        reward = float(row.get("reward") or 0.0)
        if reward == 0:
            continue

        prompt_messages = _normalize_messages(row.get("prompt_messages") or [])
        assistant_message = row.get("assistant_message") or {}
        assistant_content = str(
            assistant_message.get("content") or row.get("response_text") or ""
        ).strip()
        if not assistant_content:
            continue

        full_messages = list(prompt_messages)
        full_messages.append({"role": "assistant", "content": assistant_content})

        prompt_text = _chat_text(tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = _chat_text(tokenizer, full_messages, add_generation_prompt=False)

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        assistant_start = len(prompt_ids)
        full_ids, assistant_start = _trim_full_sequence(
            list(full_ids),
            assistant_start,
            max_length=max_length,
        )
        if len(full_ids) <= assistant_start:
            continue

        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        examples.append(
            {
                "feedback_id": int(row["feedback_id"]),
                "label": str(row.get("label") or "").strip().lower(),
                "reward": reward,
                "input_ids": list(full_ids),
                "assistant_start": int(assistant_start),
                "response_ids": list(full_ids[assistant_start:]),
                "assistant_content": assistant_content,
                "prompt_messages": prompt_messages,
                "feedback_text": (
                    row.get("feedback_text")
                    or metadata.get("feedback_text")
                    or metadata.get("sdpo_feedback_text")
                    or metadata.get("online_rl_feedback_text")
                ),
                "metadata": metadata,
            }
        )
    return examples


def _compute_ref_logprobs(sampling_client, examples: List[Dict[str, Any]]) -> None:
    """Compute reference (old policy) logprobs for each example."""
    import tinker

    for example in examples:
        prompt = tinker.ModelInput.from_ints(example["input_ids"])
        logprobs = sampling_client.compute_logprobs(prompt).result()
        example["old_logprobs"] = [lp if lp is not None else 0.0 for lp in logprobs]


def _build_importance_sampling_datum(
    example: Dict[str, Any],
    *,
    loss_fn: str,
    reward_scale: float = 1.0,
) -> "tinker.types.Datum":
    """Build a Tinker datum for the scalar-reward online RL loss."""
    import tinker
    from tinker import types

    input_ids = list(example["input_ids"])
    if len(input_ids) < 2:
        raise ValueError("Need at least two tokens to build an RL training datum")

    assistant_start = int(example["assistant_start"])
    reward = float(example["reward"]) * float(reward_scale)
    old_logprobs = list(example.get("old_logprobs") or [])

    student_input = np.array(input_ids[:-1], dtype=np.int64)
    target_tokens = np.array(input_ids[1:], dtype=np.int64)
    start_idx = max(0, assistant_start - 1)

    weights = np.zeros(len(student_input), dtype=np.float32)
    weights[start_idx:] = 1.0

    loss_fn_inputs: Dict[str, Any] = {
        "target_tokens": target_tokens,
        "weights": weights,
    }

    if loss_fn == "importance_sampling":
        ref_lps = np.array(old_logprobs[1 : 1 + len(student_input)], dtype=np.float32)
        if len(ref_lps) < len(student_input):
            ref_lps = np.pad(ref_lps, (0, len(student_input) - len(ref_lps)))

        advantages = np.zeros(len(student_input), dtype=np.float32)
        advantages[start_idx:] = reward
        loss_fn_inputs["logprobs"] = ref_lps
        loss_fn_inputs["advantages"] = advantages

    return types.Datum(
        model_input=tinker.ModelInput.from_ints(student_input.tolist()),
        loss_fn_inputs=loss_fn_inputs,
    )


def _sdpo_feedback_text(example: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Resolve the textual feedback used for SDPO reprompting."""
    explicit_feedback = str(example.get("feedback_text") or "").strip()
    if explicit_feedback:
        return explicit_feedback

    if str(example.get("label") or "").strip().lower() == "upweight":
        return str(cfg.get("sdpo_positive_feedback_fallback") or "").strip()
    return str(cfg.get("sdpo_negative_feedback_fallback") or "").strip()


def _sdpo_reprompt_messages(example: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build the feedback-conditioned teacher prompt used for SDPO."""
    prompt_messages = list(example.get("prompt_messages") or [])
    assistant_content = str(example.get("assistant_content") or "").strip()
    feedback_text = _sdpo_feedback_text(example, cfg)
    template = str(cfg.get("sdpo_reprompt_template") or "").strip()
    if not template:
        raise ValueError("sdpo_reprompt_template must not be empty")

    reprompt_content = template.format(
        prompt=_messages_as_text(prompt_messages),
        solution=assistant_content,
        feedback=feedback_text,
    )

    reprompt_messages = list(prompt_messages)
    reprompt_messages.append({"role": "assistant", "content": assistant_content})
    reprompt_messages.append({"role": "user", "content": reprompt_content})
    return reprompt_messages


def _topk_row_to_distribution(
    row: Optional[List[Tuple[int, float]]],
    *,
    actual_token: int,
    topk: int,
    weight_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one teacher top-K row into normalized token ids and weights."""
    token_ids = np.zeros(topk, dtype=np.int64)
    weights = np.zeros(topk, dtype=np.float32)

    if not row:
        token_ids[0] = int(actual_token)
        weights[0] = float(weight_scale)
        return token_ids, weights

    trimmed = list(row[:topk])
    ids = np.array([int(tok_id) for tok_id, _ in trimmed], dtype=np.int64)
    logprobs = np.array([float(logprob) for _, logprob in trimmed], dtype=np.float32)
    if len(ids) == 0:
        token_ids[0] = int(actual_token)
        weights[0] = float(weight_scale)
        return token_ids, weights

    logprobs = logprobs - np.max(logprobs)
    probs = np.exp(logprobs)
    total = float(probs.sum())
    if total <= 0:
        token_ids[0] = int(actual_token)
        weights[0] = float(weight_scale)
        return token_ids, weights

    probs = probs / total
    token_ids[: len(ids)] = ids
    weights[: len(ids)] = probs.astype(np.float32) * float(weight_scale)
    return token_ids, weights


def _build_sdpo_datum(
    teacher_sampling_client,
    example: Dict[str, Any],
    tokenizer,
    cfg: Dict[str, Any],
) -> Optional["tinker.types.Datum"]:
    """Build a top-K distillation datum for one feedback-conditioned SDPO example."""
    import tinker
    from tinker import types

    input_ids = list(example.get("input_ids") or [])
    response_ids = list(example.get("response_ids") or [])
    if len(input_ids) < 2 or not response_ids:
        return None
    if cfg.get("sdpo_require_text_feedback") and not str(example.get("feedback_text") or "").strip():
        return None

    teacher_prompt_messages = _sdpo_reprompt_messages(example, cfg)
    teacher_full_messages = list(teacher_prompt_messages)
    teacher_full_messages.append(
        {"role": "assistant", "content": str(example.get("assistant_content") or "")}
    )

    teacher_prompt_text = _chat_text(tokenizer, teacher_prompt_messages, add_generation_prompt=True)
    teacher_full_text = _chat_text(tokenizer, teacher_full_messages, add_generation_prompt=False)
    teacher_prompt_ids = tokenizer(teacher_prompt_text, add_special_tokens=False).input_ids
    teacher_full_ids = tokenizer(teacher_full_text, add_special_tokens=False).input_ids

    max_length = max(512, int(cfg.get("max_sequence_length", 4096)))
    teacher_full_ids, teacher_prompt_len = _trim_full_sequence(
        list(teacher_full_ids),
        len(teacher_prompt_ids),
        max_length=max_length,
    )
    if len(teacher_full_ids) <= teacher_prompt_len:
        return None

    topk = max(1, int(cfg.get("sdpo_topk", 100)))
    response = teacher_sampling_client.sample(
        prompt=tinker.ModelInput.from_ints(teacher_full_ids),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=topk,
    ).result()
    raw_topk = list(response.topk_prompt_logprobs or [])
    if len(raw_topk) >= len(response_ids):
        response_rows = raw_topk[-len(response_ids) :]
    else:
        response_rows = [None] * (len(response_ids) - len(raw_topk)) + raw_topk

    student_input = np.array(input_ids[:-1], dtype=np.int64)
    target_tokens = np.zeros((len(student_input), topk), dtype=np.int64)
    weights = np.zeros((len(student_input), topk), dtype=np.float32)
    start_idx = max(0, int(example["assistant_start"]) - 1)
    weight_scale = float(cfg.get("sdpo_distillation_weight", 1.0))

    for offset, actual_token in enumerate(response_ids):
        pos = start_idx + offset
        if pos >= len(student_input):
            break
        ids, probs = _topk_row_to_distribution(
            response_rows[offset] if offset < len(response_rows) else None,
            actual_token=int(actual_token),
            topk=topk,
            weight_scale=weight_scale,
        )
        target_tokens[pos] = ids
        weights[pos] = probs

    return types.Datum(
        model_input=tinker.ModelInput.from_ints(student_input.tolist()),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "weights": weights,
        },
    )


def _resolve_sdpo_teacher_sampling_client(
    service_client,
    *,
    runtime_state: Dict[str, Any],
    cfg: Dict[str, Any],
    tinker_model: str,
    existing_ckpt: str,
    save_online_rl_state,
):
    """Resolve the frozen teacher used for SDPO distillation."""
    teacher_checkpoint_path = str(runtime_state.get("sdpo_teacher_checkpoint_path") or "").strip()
    teacher_base_model = str(runtime_state.get("sdpo_teacher_base_model") or "").strip()

    if teacher_checkpoint_path.startswith("tinker://"):
        return service_client.create_sampling_client(model_path=teacher_checkpoint_path), {
            "teacher_mode": "frozen",
            "teacher_checkpoint_path": teacher_checkpoint_path,
            "teacher_base_model": teacher_base_model or tinker_model,
        }

    if teacher_base_model:
        return service_client.create_sampling_client(base_model=teacher_base_model), {
            "teacher_mode": "frozen",
            "teacher_checkpoint_path": "",
            "teacher_base_model": teacher_base_model,
        }

    if existing_ckpt.startswith("tinker://"):
        runtime_state["sdpo_teacher_checkpoint_path"] = existing_ckpt
        runtime_state.setdefault("sdpo_teacher_base_model", tinker_model)
        save_online_rl_state(runtime_state, cfg)
        return service_client.create_sampling_client(model_path=existing_ckpt), {
            "teacher_mode": "frozen",
            "teacher_checkpoint_path": existing_ckpt,
            "teacher_base_model": tinker_model,
        }

    runtime_state.setdefault("sdpo_teacher_checkpoint_path", "")
    runtime_state["sdpo_teacher_base_model"] = tinker_model
    save_online_rl_state(runtime_state, cfg)
    return service_client.create_sampling_client(base_model=tinker_model), {
        "teacher_mode": "frozen",
        "teacher_checkpoint_path": "",
        "teacher_base_model": tinker_model,
    }


def train_batch(
    export_path: str | Path,
    *,
    runtime_base_url: Optional[str] = None,
    feedback_ids: Optional[List[int]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one online-RL batch via the Tinker API."""
    import tinker
    from tinker import types

    from agent.online_rl import (
        ONLINE_RL_ALGORITHM_SDPO,
        load_online_rl_config,
        load_online_rl_state,
        normalize_online_rl_algorithm,
        publish_online_rl_adapter,
        save_online_rl_state,
    )

    cfg = cfg or load_online_rl_config()

    export_file = Path(export_path).expanduser()
    rows = [
        json.loads(line)
        for line in export_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if feedback_ids:
        wanted = {int(fid) for fid in feedback_ids}
        rows = [row for row in rows if int(row["feedback_id"]) in wanted]
    if not rows:
        raise RuntimeError(f"No trainable online-RL rows found in {export_file}")

    algorithm = normalize_online_rl_algorithm(cfg.get("algorithm"))
    tinker_model = str(cfg.get("tinker_base_model") or cfg.get("training_base_model") or "").strip()
    if not tinker_model:
        raise RuntimeError(
            "tinker_base_model (or training_base_model) is required for Tinker training"
        )

    lora_rank = int(cfg.get("tinker_lora_rank") or cfg.get("lora_rank", 32))
    binary_loss_fn = str(cfg.get("tinker_loss_fn") or "importance_sampling").strip().lower() or "importance_sampling"
    if algorithm == ONLINE_RL_ALGORITHM_SDPO:
        binary_loss_fn = "importance_sampling"

    learning_rate = float(cfg.get("learning_rate", 2e-6))
    weight_decay = float(cfg.get("weight_decay", 0.1))
    configured_total_steps = int(cfg.get("train_steps", 16))
    grad_accum = max(1, int(cfg.get("gradient_accumulation_steps", 4)))

    api_key = str(cfg.get("tinker_api_key") or "").strip()
    if not api_key:
        raise RuntimeError(
            "TINKER_API_KEY environment variable or tinker_api_key config is required"
        )
    os.environ.setdefault("TINKER_API_KEY", api_key)

    logger.info(
        "Connecting to Tinker API for model %s (rank=%d, mode=%s)",
        tinker_model,
        lora_rank,
        algorithm,
    )
    service_client = tinker.ServiceClient()

    runtime_state = load_online_rl_state(cfg)
    existing_ckpt = str(
        runtime_state.get("tinker_checkpoint_path") or cfg.get("tinker_checkpoint_path") or ""
    ).strip()
    if existing_ckpt and existing_ckpt.startswith("tinker://"):
        logger.info("Resuming from checkpoint: %s", existing_ckpt)
        try:
            training_client = service_client.create_training_client_from_state(existing_ckpt)
        except Exception as exc:
            logger.warning(
                "Failed to resume from checkpoint %s: %s. Starting fresh.",
                existing_ckpt,
                exc,
            )
            training_client = service_client.create_lora_training_client(
                base_model=tinker_model,
                rank=lora_rank,
            )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=tinker_model,
            rank=lora_rank,
        )

    tokenizer = training_client.get_tokenizer()
    examples = _build_tinker_examples(rows, tokenizer, cfg)
    if not examples:
        raise RuntimeError("No usable assistant trajectories found in the exported feedback batch")

    if algorithm == ONLINE_RL_ALGORITHM_SDPO and configured_total_steps <= 0:
        total_steps = len(examples)
    else:
        total_steps = max(1, configured_total_steps)

    need_rl = float(cfg.get("sdpo_rl_weight", 1.0)) > 0.0 or algorithm != ONLINE_RL_ALGORITHM_SDPO
    if need_rl and binary_loss_fn in {"importance_sampling", "cispo", "ppo"}:
        logger.info("Computing reference logprobs for %d examples...", len(examples))
        ref_sampling_client = training_client.save_weights_and_get_sampling_client(name="ref_policy")
        _compute_ref_logprobs(ref_sampling_client, examples)
    else:
        ref_sampling_client = None

    teacher_info: Dict[str, Any] = {
        "teacher_mode": "",
        "teacher_checkpoint_path": "",
        "teacher_base_model": "",
    }
    sdpo_example_count = 0
    text_feedback_count = 0
    if algorithm == ONLINE_RL_ALGORITHM_SDPO:
        teacher_sampling_client, teacher_info = _resolve_sdpo_teacher_sampling_client(
            service_client,
            runtime_state=runtime_state,
            cfg=cfg,
            tinker_model=tinker_model,
            existing_ckpt=existing_ckpt,
            save_online_rl_state=save_online_rl_state,
        )
        logger.info(
            "Building frozen-teacher SDPO targets for %d examples (topk=%d)...",
            len(examples),
            int(cfg.get("sdpo_topk", 100)),
        )
        for example in examples:
            if str(example.get("feedback_text") or "").strip():
                text_feedback_count += 1
            example["sdpo_datum"] = _build_sdpo_datum(
                teacher_sampling_client,
                example,
                tokenizer,
                cfg,
            )
            if example["sdpo_datum"] is not None:
                sdpo_example_count += 1

    logger.info(
        "Starting training: %d steps, %d examples, grad_accum=%d",
        total_steps,
        len(examples),
        grad_accum,
    )
    t0 = time.time()
    stats: Dict[str, Any] = {
        "steps": total_steps,
        "used_examples": len(examples),
        "masked_updates": 0,
        "applied_updates": 0,
        "mean_reward": sum(ex["reward"] for ex in examples) / float(len(examples)),
        "binary_loss_fn": binary_loss_fn,
        "sdpo_teacher_mode": teacher_info.get("teacher_mode"),
        "sdpo_teacher_checkpoint_path": teacher_info.get("teacher_checkpoint_path"),
        "sdpo_teacher_base_model": teacher_info.get("teacher_base_model"),
        "sdpo_topk": int(cfg.get("sdpo_topk", 100)),
        "sdpo_examples": sdpo_example_count,
        "text_feedback_examples": text_feedback_count,
    }

    loss_fn_config: Dict[str, float] = {}
    if binary_loss_fn in {"ppo", "cispo"}:
        loss_fn_config = {"clip_low_threshold": 0.2, "clip_high_threshold": 0.2}

    rl_weight = float(cfg.get("sdpo_rl_weight", 1.0))
    for step in range(total_steps):
        example = examples[step % len(examples)]

        if need_rl:
            rl_datum = _build_importance_sampling_datum(
                example,
                loss_fn=binary_loss_fn,
                reward_scale=rl_weight,
            )
            training_client.forward_backward(
                [rl_datum],
                binary_loss_fn,
                loss_fn_config or None,
            ).result()

        if algorithm == ONLINE_RL_ALGORITHM_SDPO and example.get("sdpo_datum") is not None:
            training_client.forward_backward(
                [example["sdpo_datum"]],
                "cross_entropy",
                None,
            ).result()

        if (step + 1) % grad_accum == 0 or (step + 1) == total_steps:
            training_client.optim_step(
                types.AdamParams(
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                )
            ).result()
            stats["applied_updates"] += 1

        if (step + 1) % max(1, total_steps // 4) == 0:
            logger.info("Step %d/%d completed", step + 1, total_steps)

    elapsed = time.time() - t0
    logger.info("Training completed in %.1fs", elapsed)

    adapter_name = str(cfg.get("adapter_name") or "hermes-online-rl").strip()
    ckpt_name = f"{adapter_name}_{int(time.time())}"
    save_result = training_client.save_weights_for_sampler(name=ckpt_name).result()
    checkpoint_path = save_result.path
    logger.info("Saved Tinker checkpoint: %s", checkpoint_path)

    published_state = publish_online_rl_adapter(
        checkpoint_path,
        runtime_base_url=runtime_base_url,
        base_model=tinker_model,
        cfg=cfg,
    )

    stats.update(
        {
            "adapter_dir": checkpoint_path,
            "tinker_checkpoint_path": checkpoint_path,
            "published_model": published_state.get("active_model_name"),
            "backend": "tinker",
            "algorithm": algorithm,
            "tinker_base_model": tinker_model,
            "elapsed_seconds": elapsed,
        }
    )
    return stats
