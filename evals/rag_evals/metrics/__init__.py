"""Metrics for evals."""

import os

metrics = []

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

for file in os.listdir(PROMPTS_DIR):
    if file.endswith(".md"):
        with open(os.path.join(PROMPTS_DIR, file), "r", encoding="utf-8") as prompt_file:
            metrics.append({"name": file.replace(".md", ""), "prompt": prompt_file.read()})
