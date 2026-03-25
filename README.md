# Hermes Agent Online RL

**Self-improving AI through human feedback — train LoRA adapters live as you use [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Every time you interact with Hermes Agent, you're rolling out a trajectory from your local model. This project adds a tight feedback loop: after each response, you rate it (upweight / downweight / skip), and those signals continuously train a LoRA adapter using **MIS-PO** — the same RL algorithm behind [Step 3.5 Flash](https://arxiv.org/abs/2602.10604). Your model gets better at *your* workflows, on *your* hardware, without ever sending data to a remote server.

---

## Why Online RL on Hermes Agent?

Traditional RLHF requires collecting large datasets, training offline, and deploying new checkpoints. Online RL collapses this into a single continuous loop:

| Traditional RLHF | Hermes Online RL |
|---|---|
| Collect thousands of preference pairs | Rate responses as you use them |
| Train offline on a GPU cluster | LoRA trains locally in the background |
| Deploy a new model checkpoint | Adapter hot-loads into your running server |
| Generic model for all users | Personalized to your exact workflows |
| Weeks of iteration | Minutes between feedback and improvement |

### What This Means in Practice

- **Code assistant that learns your style** — Upweight responses that follow your conventions, downweight ones that don't. After a few sessions, the model defaults to your preferred patterns.
- **Research workflows** — If you use Hermes for literature review, data analysis, or writing, the model learns what level of detail you want, which tools you prefer, and how you like results formatted.
- **Agentic tasks** — When Hermes plans multi-step terminal commands, file operations, or web research, your feedback teaches it which strategies work for your environment.
- **Domain specialization** — Working in a specific codebase, language, or field? The LoRA adapter accumulates domain knowledge from your corrections.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hermes Agent CLI                          │
│                                                             │
│  User prompt ──► Local Model (vLLM/Ollama/MLX) ──► Response │
│                         │                          │        │
│                    LoRA Adapter                     │        │
│                    (hot-loaded)              ┌──────┴──────┐ │
│                         ▲                   │  Feedback UI │ │
│                         │                   │  [1] ⬆ Up    │ │
│                         │                   │  [2] ⬇ Down  │ │
│                         │                   │  [3] ⊘ Skip  │ │
│                         │                   └──────┬──────┘ │
│                         │                          │        │
│                   ┌─────┴──────┐                   │        │
│                   │  MIS-PO    │◄──────────────────┘        │
│                   │  Trainer   │                             │
│                   │ PyTorch/MLX│  Fires after min_batch_size │
│                   └────────────┘  feedback samples (def: 8)  │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Backend Training

The trainer automatically selects the best backend for your hardware:

| Hardware | Backend | How It Works |
|---|---|---|
| **Apple Silicon (M1–M5)** | MLX | Native Metal acceleration via `mlx` + `mlx-lm`. Zero-copy unified memory. |
| **NVIDIA GPU** | PyTorch (CUDA) | Standard PyTorch with `torch.cuda`. BF16/FP16 auto-selected. |
| **Apple (no MLX installed)** | PyTorch (MPS) | Falls back to Metal Performance Shaders via PyTorch. |
| **CPU** | PyTorch (CPU) | Works everywhere, just slower. FP32. |

**Auto-detection is the default.** Set `device: auto` (or omit it) and the trainer picks the right backend. You can override with `device: mlx`, `device: cuda`, `device: mps`, or `device: cpu`.

### The Feedback Loop

1. **You chat with Hermes** using any workflow — coding, research, writing, tool use
2. **After each response**, a compact feedback panel appears with three options:
   - **Upweight** (`1` or `↑/Enter`) — reward = +1.0, "learn from this"
   - **Downweight** (`2` or `↓/Enter`) — reward = -1.0, "unlearn this"
   - **Skip** (`3` or `↓↓/Enter`) — no training signal
3. **Feedback accumulates** in a local SQLite database with full trajectory context
4. **When `min_batch_size` samples are collected** (default: 8), the MIS-PO trainer fires automatically in the background
5. **The trained LoRA adapter is hot-loaded** into your running vLLM or Ollama server (or saved to disk for MLX)
6. **Next response uses the updated adapter** — the loop continues

---

## MIS-PO: The RL Algorithm

This implementation uses **Metropolis Independence Sampling - Filtered Policy Optimization (MIS-PO)**, the same algorithm introduced in the [Step 3.5 Flash technical report](https://arxiv.org/abs/2602.10604). MIS-PO is specifically designed for stable RL on language models without requiring grouped rollouts.

### Why MIS-PO over PPO/GRPO?

| Algorithm | Requires | Issue for Online RL |
|---|---|---|
| PPO | Grouped rollouts, critic network | Too heavy — needs multiple responses per prompt, doubles VRAM for critic |
| GRPO | Multiple rollouts per prompt | Can't do this in real-time — user only sees one response |
| REINFORCE | Nothing extra | High variance, unstable |
| **MIS-PO** | **Single trajectory + binary filtering** | **Perfect for online RL — one response = one trajectory** |

### The Objective

MIS-PO uses a REINFORCE-style policy gradient with binary masking instead of continuous importance weighting:

```
L_actor = -E[I(τ) · log π_θ(a_t|s_t) · Â_t]
```

Where `I(τ)` is a binary indicator that filters off-policy samples at two granularities:

**Token-level filtering:** For each token, compute the probability ratio between the training policy and the inference policy:

```
x_t = π_θ_train(a_t|s_t) / π_θ_inference(a_t|s_t)
```

Only tokens where `ρ_min ≤ x_t ≤ ρ_max` contribute to the gradient. Default bounds: `[0.8, 1.25]`.

**Trajectory-level filtering:** Compute the geometric mean of token ratios across the full assistant response:

```
ρ̄(τ) = (∏_t x_t)^(1/T)
```

The entire trajectory is rejected if `ρ̄(τ)` falls outside `[0.9, 1.1]`. This prevents accumulated small token-level drifts from corrupting the gradient.

### KL Regularization

An unbiased KL divergence penalty prevents the adapter from drifting too far from the base model:

```
L_KL = E[exp(log_ratio) - log_ratio - 1]
```

With coefficient `β = 0.001` (matching Step 3.5 Flash). This is the Schulman k3 estimator — unbiased and low-variance.

**VRAM impact of KL: zero.** The KL term reuses tensors already computed for the actor loss (`new_logprobs`, `old_logprobs`). No additional forward pass or model copy needed.

### Why Binary Filtering > Clipping

PPO clips the importance ratio to `[1-ε, 1+ε]`, which still passes scaled gradients from off-policy tokens. MIS-PO completely excludes off-policy tokens, treating the remaining ones as on-policy. This:

- Reduces gradient variance (empirically 5x lower than PPO per Step 3.5 Flash)
- Is more stable for long trajectories where small per-token errors compound
- Works with a single trajectory (no need for the "group" in GRPO)

---

## LoRA Training Details

### Why LoRA?

Full fine-tuning would require storing optimizer states for every parameter — 2-4x the model's VRAM. LoRA trains only low-rank adapter matrices (~0.1-1% of parameters), keeping VRAM overhead minimal:

| Component | VRAM |
|---|---|
| Base model (frozen) | Same as inference |
| LoRA adapter parameters | ~10-50 MB |
| LoRA gradients + optimizer states | ~30-150 MB |
| One forward/backward pass activations | Same as inference |
| **Total training overhead** | **~50-200 MB above inference** |

Since you're already running the model locally for inference, training adds minimal extra VRAM.

### MLX vs PyTorch LoRA

Both backends implement identical LoRA training with the same MIS-PO objective. The difference is the underlying framework:

| Aspect | PyTorch Backend | MLX Backend |
|---|---|---|
| Library | `torch` + `peft` | `mlx` + `mlx-lm` |
| Adapter format | PEFT safetensors | mlx-lm `adapters.safetensors` |
| LoRA application | `peft.get_peft_model()` | `mlx_lm.tuner.linear_to_lora_layers()` |
| Autodiff | `loss.backward()` | `mlx.nn.value_and_grad()` |
| Optimizer | `torch.optim.AdamW` | `mlx.optimizers.AdamW` |
| Memory model | Separate CPU/GPU memory | Unified memory (zero-copy) |
| Quantized training | Manual setup | Automatic QLoRA when loading 4-bit models |

**Apple Silicon advantage:** MLX leverages unified memory, meaning the model, optimizer states, and activations share the same memory pool — no CPU↔GPU copies. A 7B model that needs 14GB for inference might only need ~14.5GB total for inference + LoRA training.

### Default Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| LoRA rank | 16 | Good balance of capacity vs. VRAM |
| LoRA alpha | 32 | 2x rank (standard scaling) |
| LoRA dropout | 0.05 | Light regularization |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | All attention + MLP projections |
| Learning rate | 2e-6 | Conservative for stability |
| Weight decay | 0.1 | AdamW regularization |
| Warmup steps | 20 | Gradual LR ramp |
| Train steps per batch | 16 | Steps per training trigger |
| Gradient accumulation | 4 | Effective batch = 4 examples |
| Max sequence length | 262144 (256K) | Truncates longer trajectories |
| Grad norm clip | 1.0 | Prevents exploding gradients |
| Min batch size | 8 | Feedback samples before training fires |
| Max saved adapters | 4 | Keeps last 4, auto-cleans older ones |

### Adapter Lifecycle

1. **First training:** Creates a fresh LoRA adapter from the base model
2. **Subsequent trainings:** Loads the previous adapter and continues training (warm-start)
3. **Hot-loading:** After training, the adapter is published to vLLM (`/v1/load_lora_adapter`), Ollama (`ollama create`), or saved to disk (MLX)
4. **Cleanup:** Only the last `max_saved_adapters` checkpoints are kept on disk

---

## Installation & Setup

### Prerequisites

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) installed and working
- A local model server (vLLM, Ollama, llama.cpp, LM Studio) — or MLX for Apple Silicon native inference
- Python 3.11+

### Install

**For NVIDIA GPU / CPU (PyTorch backend):**

```bash
pip install -e ".[online-rl]"
```

This installs `torch`, `transformers`, and `peft`.

**For Apple Silicon (MLX backend):**

```bash
pip install -e ".[online-rl-mlx]"
```

This installs `mlx` and `mlx-lm`. The trainer auto-detects Apple Silicon and uses MLX natively.

**For both backends (maximum compatibility):**

```bash
pip install -e ".[online-rl,online-rl-mlx]"
```

### Configure

```bash
# Interactive setup wizard
hermes setup online-rl

# Or configure manually
hermes config set online_rl.enabled true
hermes config set online_rl.prompt_after_response true
hermes config set online_rl.training_base_model "your-model-name"
hermes config set online_rl.local_only true
```

**MLX-specific example:**

```yaml
# ~/.hermes/config.yaml
online_rl:
  enabled: true
  prompt_after_response: true
  training_base_model: mlx-community/Mistral-7B-Instruct-v0.3-4bit
  device: auto  # auto-detects MLX on Apple Silicon
```

> **Tip:** Use quantized models from the [mlx-community](https://huggingface.co/mlx-community) on Hugging Face for best MLX performance. 4-bit models automatically get QLoRA — the base stays quantized while only LoRA weights train in full precision.

The setup wizard will:
1. Scan for local inference servers (vLLM, Ollama, llama.cpp, LM Studio)
2. Auto-detect available models
3. Let you configure LoRA hyperparameters
4. Write everything to `~/.hermes/config.yaml`

### Use

```bash
hermes  # start chatting — feedback panel appears after each response
```

Press `1` to upweight, `2` to downweight, `3` to skip. After 8 feedback samples, training fires automatically.

### Monitor

```bash
hermes online-rl status          # Show config and active adapter
hermes online-rl list-feedback   # List stored feedback
hermes online-rl list-feedback --pending  # Only untrained samples
```

---

## UI

### Feedback Panel

After each response, a compact panel appears:

```
╭─ Online RL ──────────────── 42s ─╮
│                                   │
│  ▲/▼ navigate  Enter confirm     │
│                                   │
│  ❯ ⬆ Upweight   learn from this  │
│    ⬇ Downweight  unlearn this     │
│    ⊘ Skip        no training      │
│                                   │
╰──────────────────────────────────╯
```

- **Countdown timer** in the title bar (auto-skips after timeout)
- **Number keys** `1`/`2`/`3` for instant selection
- **Arrow keys + Enter** for navigation
- **Ctrl+C** to skip

### Status Bar

When online RL is active, a green `RL` indicator appears in the bottom status bar alongside the model name and context usage.

---

## Configuration Reference

All settings live under `online_rl:` in `~/.hermes/config.yaml`:

```yaml
online_rl:
  enabled: true
  prompt_after_response: true
  local_only: true                    # Only activate for localhost servers
  training_base_model: "your-model"   # HuggingFace model name or path

  # MIS-PO bounds (matching Step 3.5 Flash)
  token_ratio_min: 0.8
  token_ratio_max: 1.25
  trajectory_ratio_min: 0.9
  trajectory_ratio_max: 1.1
  kl_coefficient: 0.001

  # LoRA
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

  # Training
  learning_rate: 0.000002
  weight_decay: 0.1
  warmup_steps: 20
  train_steps: 16
  gradient_accumulation_steps: 4
  min_batch_size: 8
  max_batch_size: 64
  max_sequence_length: 262144

  # Adapter management
  adapter_name: hermes-online-rl
  max_saved_adapters: 4
  builtin_trainer: true

  # Device — auto-detects MLX on Apple Silicon, CUDA on NVIDIA, etc.
  device: auto          # auto, mlx, cuda, mps, cpu
  torch_dtype: auto     # auto, bf16, fp16, fp32 (PyTorch backend only)
```

Environment variable overrides are available for all settings (e.g., `HERMES_ONLINE_RL_ENABLED=true`).

---

## Files

| File | Purpose |
|---|---|
| `agent/online_rl.py` | Feedback capture, backend detection, adapter publishing (vLLM/Ollama/MLX/Tinker) |
| `agent/online_rl_trainer.py` | PyTorch MIS-PO trainer with backend routing |
| `agent/online_rl_trainer_mlx.py` | MLX-native MIS-PO trainer for Apple Silicon |
| `agent/online_rl_trainer_tinker.py` | Tinker API backend for hosted SOTA model training |
| `cli.py` | Feedback panel UI with icons, countdown, hotkeys, RL status bar |
| `hermes_cli/setup.py` | `setup_online_rl()` wizard with server auto-detection |
| `hermes_cli/main.py` | `hermes online-rl` subcommands |
| `hermes_state.py` | RL feedback SQLite schema and queries |
| `pyproject.toml` | `[online-rl]`, `[online-rl-mlx]`, and `[online-rl-tinker]` optional dependency groups |

---

## How It Works End-to-End

1. You run `hermes` and chat normally
2. Your local model (via vLLM/Ollama/MLX) generates a response — this is a **trajectory rollout**
3. The feedback panel appears — you press `1` (good), `2` (bad), or `3` (skip)
4. Feedback is stored in SQLite with the full message context (prompt + response)
5. When 8+ rated samples accumulate, `maybe_trigger_online_rl_training()` fires
6. The trainer spawns as a background subprocess:
   - **Auto-selects backend:** Tinker when `TINKER_API_KEY` + `tinker_base_model` are configured, otherwise MLX on Apple Silicon, PyTorch+CUDA on NVIDIA, PyTorch+MPS/CPU elsewhere
   - Loads the base model + existing LoRA adapter (if any)
   - Computes reference logprobs (π_inference)
   - Runs 16 MIS-PO training steps with binary filtering
   - Saves the updated adapter to `~/.hermes/online_rl/adapters/` or publishes the Tinker checkpoint metadata
   - Hot-loads it into the running inference server (or saves to disk for MLX)
7. Your next conversation uses the improved adapter
8. The cycle repeats — your model continuously improves at your specific workflows

---

## Tinker Backend

The online RL stack now supports a hosted Tinker backend in addition to local PyTorch and MLX training. That gives you a path to train adapters against frontier base models through Tinker while keeping the same Hermes feedback loop and online adaptation flow.

Use the Tinker path when you want:

- A hosted training backend instead of running local fine-tuning
- Access to frontier base models such as Kimi-class models
- The same Hermes feedback capture flow with Tinker-managed checkpoints

Configure it with:

- `TINKER_API_KEY`
- `WANDB_API_KEY`
- `online_rl.tinker_base_model`
- `online_rl.training_backend: tinker` (or leave backend on `auto` and provide the Tinker config)

---

## References

- [Step 3.5 Flash Technical Report](https://arxiv.org/abs/2602.10604) — MIS-PO algorithm
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — Base agent framework
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Trust Region Masking](https://arxiv.org/abs/2512.23075) — Theoretical basis for sequence-level trust regions
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLM fine-tuning and inference on MLX

---

## License

MIT — same as Hermes Agent.
