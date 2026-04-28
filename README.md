# Truthful Resume Optimizer Agent (Local LLM)

## Overview
This project is a Python-based AI agent that improves resume content using a **local LLM (Llama via Ollama)** while ensuring the output remains truthful and does not introduce fake experience.

The agent:
- Compares a resume with a job description
- Calculates an **ATS-style keyword alignment score**
- Rewrites resume bullets to better match the job description
- Shows **before vs after score improvement**

---

## Key Features

- 📊 ATS-style keyword alignment scoring (before & after)
- 🤖 Local LLM-based resume rewriting (no paid API required)
- 🔒 Controlled output to prevent hallucination or fake experience
- 🧠 Hybrid architecture (LLM + rule-based scoring)
- 📝 Structured, professional resume bullet output
- 📉 Gap analysis (missing/weak keywords)

---

## Why This Project

Most resume tools either:
- blindly rewrite content (risking fake claims), or  
- rely on simple keyword matching  

This project combines:
- **LLM intelligence** → to understand job requirements  
- **Python logic** → to enforce scoring and truthfulness  

---

## Tech Stack

- Python
- Ollama (local LLM runtime)
- Llama 3.2 (1B or 3B model)
- Regex (text preprocessing)
- JSON parsing

---

## How It Works

1. User inputs:
   - Job Description
   - Resume text

2. The agent:
   - Extracts important requirements using LLM
   - Assigns weights to categories (skills, tools, responsibilities)
   - Calculates **original match score**
   - Rewrites resume using controlled LLM prompt
   - Calculates **optimized match score**

3. Outputs:
   - Before/After score comparison
   - Missing keywords (before & after)
   - Improved resume bullets
   - Rewrite review

---

## Setup Instructions

### 1. Install Ollama
Download from:
https://ollama.com

---

### 2. Start Ollama server
```bash
ollama serve
```

### 3. Pull model
```bash
ollama pull llama3.2:1b
```

### 4. Install Python dependency
```bash
pip install ollama
```

### 5. Run the project
```bash
python3 day6_resume_optimizer.py
```
