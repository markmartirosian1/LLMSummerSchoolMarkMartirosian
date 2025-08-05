# ArmLLM 2025 — Summer Program Exercises

This repository contains code and notebooks developed as part of the **2025 ArmLLM Summer School**, a hands-on training program in large language models, multimodality, agentic workflows and more. It was a six day program, each of which was based on a topic designed to highlight a key component in building and evaluating modern LLM systems.

---

## 📁 Directory Structure

```
2025/
├── Intro & Pretraining/               # Exercise 1
├── Multimodality/                     # Exercise 2
├── Post-Training/                     # Exercise 3
├── Inference-Timing/ (notebooks only) # Exercise 4
├── Agents/                            # Exercise 5
├── Security/ (notebooks only)         # Exercise 6
```

Each subfolder includes implementation code, models, and notebooks. Exercises 4 and 6 are provided as annotated Jupyter notebooks.

---

## 🧠 Exercise 1 — Transformer Pretraining ("Module 1")

**Location**: `Module 1: Intro & Pretraining/`

This module covers LLM pretraining fundamentals. Implemented a mini-transformer and a LLaMA-style text generation pipeline from scratch.

**Core Files:**

* `transformer.py`: Implements attention, feed-forward blocks, layer norm, etc.
* `llama_generation.py`: Sampling from a trained language model.

**Key Concepts:**

* Positional embeddings
* Autoregressive decoding
* Multi-head attention

---

## 🖼️ Exercise 2 — Vision-Language Alignment ("Multimodality")

**Location**: `Multimodality/`

Implements a VLM pipeline using:

* SigLIP (image encoder)
* Qwen2 (language decoder)

**Core File:** `vlm_unsolved.py`

**Tasks:**

* Combine vision + language embeddings
* Align image captions and token outputs

---

## 📈 Exercise 3 — Post-Training with GRPO ("Post-Training")

**Location**: `Post-Training/`

Fine-tuning a base LLM using preference-based optimization. We apply Generalized Reinforcement Policy Optimization (GRPO) with custom reward shaping.

**Core Files:**

* `train_grpo.py`: Implements GRPO with logging and batching
* `utils.py`: Reward shaping, normalization

**Key Concepts:**

* Preference modeling
* Off-policy optimization
* Sampling-based training loops

---

## ⏱️ Exercise 4 — Inference Efficiency & Prompt Strategies

**Location**: Inference-Timing/

This notebook explored inference-time tradeoffs using vLLM with multiple decoding strategies.

**Key Techniques:**

* Majority voting vs Best-of-N
* Prompt editing to improve consistency
* Token-length vs latency analysis

**Model:** Mistral-7B via vLLM API

---

## 🤖 Exercise 5 — LLM Agents and Tool Use ("Agents")

**Location**: `Agents/`

Participants built various tool-using agents:

### Tool-Calling Agent

**Folder**: `Agents/mcp/tool_calling/`

* Uses Pydantic schemas to format inputs/outputs
* Chains tools like summary generation, translation, and email
* Supports majority vote & feedback-refinement loop

### Memory Chat Agent

**Folder**: `Agents/mcp/custom_servers/weather/`

* Conversational memory + weather lookup
* JSON backend and persistent history

### Web Browser Agent

**Folder**: `Agents/mcp/web_agent/`

* Simulates scraping web pages via JSON-based interactions

**Notebook Add-on**: Includes a functional tool-calling demo for building autonomous task agents.

**Key Concepts:**

* Function calling
* Multi-tool agents
* Self-evaluation & retry logic

---

## 🛡️ Exercise 6 — Adversarial Robustness & LLM Security

**Location**: Security

Focused on vision-language adversarial attacks on CLIP-like models:

**Attack Types:**

* FGSM, PGD (L2 & L∞ norm)
* Targeted vs untargeted attacks
* Transferability from proxy models
* Sticker-based misclassification (physical adversaries)

**Metrics:**

* Pre/post-attack classification accuracy
* Visualization of perturbed samples

---

## ✍️ Credits

Developed during the ArmLLM 2025 Summer School.

