# Prompt Framework + LLM Evaluation Harness

## Executive Summary
Most prompt engineering work breaks silently when models change. This repo provides a **full, production-grade prompt evaluation framework** with:
- Modular prompt templates  
- Structured-output schemas  
- Built-in safety checks  
- Deterministic evaluation runner  
- HTML/Markdown reports  
- Streamlit front-end UI for non-technical users  

Designed for modern LLM teams, this project enables **repeatable, measurable, and safe** prompt engineering workflows.

## Why This Matters
Organizations today expect:
- Evaluations that detect **drift** across model versions  
- **Hallucination** measurement  
- Automated **instruction-following** checks  
- **Safety** screening for self-harm, violence, illegal activity, and dangerous medical behavior  
- A frontend interface for non-engineers to test prompts  

This repo demonstrates these capabilities end-to-end.

## Architecture Diagram

```
                 ┌───────────────────────┐
                 │    Prompt Library      │
                 │  (templates + schemas) │
                 └──────────┬────────────┘
                            │
                            ▼
                 ┌───────────────────────┐
                 │   Evaluation Dataset   │
                 │ (inputs + gold labels) │
                 └──────────┬────────────┘
                            │
                            ▼
                 ┌───────────────────────┐
                 │     Eval Runner        │
                 │ (LLM calls + scoring)  │
                 └──────────┬────────────┘
                            │
                            ▼
                 ┌─────────────────────────┐
                 │    Report Generator      │
                 │ (CSV + Markdown + HTML)  │
                 └──────────┬──────────────┘
                            │
                            ▼
                 ┌─────────────────────────┐
                 │     Streamlit UI         │
                 │ (Non-technical users)    │
                 └──────────────────────────┘
```

## Features

### Prompt Library
- Classification, extraction, and summarization templates  
- Pydantic schemas enforcing strict JSON outputs  
- Guardrail-language included by default  

### Evaluation Dataset
- 25–100 curated examples  
- Includes adversarial, ambiguous, and edge cases  
- Gold-standard outputs for deterministic scoring  

### Scoring System
- Structure compliance  
- Faithfulness (token overlap proxy)  
- Completeness & accuracy  
- Deviation + drift detection  

### Eval Runner
Run the entire eval suite:

```bash
python -m src.llm_eval_suite.eval.run_eval --config src/llm_eval_suite/config/default_config.yaml
```

### Safety Engine
Multi-layer safety system:
1. Regex-based risk detection  
2. Sentiment scoring  
3. Semantic similarity risk modeling (OpenAI embeddings)  
4. Combined input + output safety analysis  

Flags include:
- Self-harm content  
- Violence  
- Illegal activity  
- Dangerous medical behavior  
- Severe negative emotional distress  

### Streamlit Web UI
Run:

```bash
python -m streamlit run app/streamlit_app.py
```

Capabilities:
- Upload CSV / JSONL or paste text  
- Select task (summarization, extraction, classification)  
- See:
  - Parsed JSON  
  - Safety flags  
  - Sentiment  
  - Semantic risk matches  
  - Faithfulness score  

Perfect for non-technical stakeholders.

## Results & Metrics (Sample)
_Local test results shown as example._

| Metric | Baseline | Improved |
|--------|----------|----------|
| Structured-output accuracy | 63% | **94%** |
| Hallucination rate | 22% | **4%** |
| Safety detection recall | 41% | **92%** |
| Drift stability across 3 models | 48% | **90%** |

## How to Use

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
Create `.env`:
```
OPENAI_API_KEY=your_key_here
MODEL_NAME=gpt-4.1
MODEL_TEMPERATURE=0.0
```

### 3. Run evaluations (CLI)
```bash
python -m src.llm_eval_suite.eval.run_eval --config src/llm_eval_suite/config/default_config.yaml
```

### 4. Launch Streamlit UI
```bash
python -m streamlit run app/streamlit_app.py
```

## How to Extend

### Add New Task
1. Create a new prompt in `src/.../prompts/`  
2. Add a schema in `schemas/`  
3. Add examples in `data/`  
4. Add scoring logic in `scoring.py`

### Add New Model Backend
Implement a client in:
```
src/llm_eval_suite/utils/llm_client.py
```

### Add New Metrics
Modify:
```
src/llm_eval_suite/eval/scoring.py
```

## Limitations & Future Work
- Add automated CI (GitHub Actions) to run evals on PR  
- Support for Gemini and Bedrock models  
- Automatic adversarial test generation  
- Advanced summarization metrics (ROUGE, BERTScore)  
- Dedicated safety benchmark suite  

## Project Metadata
- **Author:** Moses Shenassa  
- **Tags:** llm, evaluation, prompt-engineering, ai-safety, python  
- **Repo:** https://github.com/moses-shenassa/llm-prompt-framework-and-eval-suite
