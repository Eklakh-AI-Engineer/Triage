"""Unified LLM client (audit fixes: no stateful provider mutation, Ollama timeout,
honest token budgets, explicit JSON mode)."""
import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_TIMEOUT_S = 120.0

class UnifiedLLMClient:
    """Unified LLM interface supporting Hugging Face Inference API with fallback.

    Fixes applied :
    - A transient HF failure no longer permanently mutates the instance; the
      fallback is scoped to the failing call.
    - Ollama calls carry an explicit timeout.
    - `max_tokens` is honored (no silent 768 cap).
    - JSON mode is an explicit parameter, not substring sniffing.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        from src.config import (
            LLM_PROVIDER, LLM_HF_MODEL, JUDGE_HF_MODEL, JUDGE_OLLAMA_MODEL,
            LLM_MODEL, HF_TOKEN,
        )

        self.provider = (provider or os.getenv("LLM_PROVIDER") or LLM_PROVIDER).lower()
        # Judge-specific provider override: run the generator fully local
        # (e.g., ollama on a small GPU) while the judge uses a stronger
        # remote model (e.g., huggingface cloud = Qwen2.5-72B-Instruct).
        # An explicit provider kwarg always wins over the env override.
        if not provider and (model == JUDGE_HF_MODEL or model == "judge"):
            judge_provider = (os.getenv("JUDGE_PROVIDER") or "").strip().lower()
            if judge_provider:
                self.provider = judge_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token = os.getenv("HF_TOKEN") or HF_TOKEN
        self.ollama_model = (
            JUDGE_OLLAMA_MODEL
            if model == JUDGE_HF_MODEL
            else LLM_MODEL
            if model is None or model in (LLM_MODEL, LLM_HF_MODEL)
            else model
        )

        if self.provider == "nvidia":
            from src.config import NVIDIA_API_KEY, NVIDIA_BASE_URL, LLM_NVIDIA_MODEL, JUDGE_NVIDIA_MODEL
            self.nvidia_key = os.getenv("NVIDIA_API_KEY") or NVIDIA_API_KEY
            self.nvidia_url = (os.getenv("NVIDIA_BASE_URL") or NVIDIA_BASE_URL).rstrip("/") + "/chat/completions"
            if model == JUDGE_HF_MODEL or model == "judge":
                self.model = JUDGE_NVIDIA_MODEL
            elif not model or model in (LLM_MODEL, LLM_HF_MODEL, "llama3.1:8b"):
                self.model = LLM_NVIDIA_MODEL
            else:
                self.model = model
            if not self.nvidia_key:
                logger.warning("NVIDIA_API_KEY not found; falling back to Ollama.")
                self.provider = "ollama"
                self.model = self.ollama_model
            else:
                return

        if self.provider == "huggingface" and self.token:
            self.model = LLM_HF_MODEL if (not model or model == LLM_MODEL) else model
            try:
                from huggingface_hub import InferenceClient
                self._hf_client = InferenceClient(api_key=self.token, timeout=90.0)
                return
            except Exception as e:
                logger.warning("Failed to initialize HF client: %s. Using Ollama.", e)
        else:
            if self.provider != "nvidia":
                logger.info("HF provider unavailable (no token) - using Ollama.")
        self.provider = "ollama"
        self.model = self.ollama_model

    def _chat_ollama(self, messages, temp, tokens, json_mode, model):
        import ollama
        kwargs = {"options": {"temperature": temp, "num_predict": tokens}}
        if json_mode:
            kwargs["format"] = "json"
        if model.startswith("qwen3:"):
            kwargs["think"] = False
        try:
            res = ollama.chat(model=model, messages=messages, **kwargs)
        except TypeError:
            kwargs.pop("think", None)
            res = ollama.chat(model=model, messages=messages, **kwargs)
        return res.message.content or ""

    def _chat_ollama_http(self, messages, temp, tokens, json_mode, model):
        """Timeout-guarded HTTP call (the ollama pkg has no timeout kwarg)."""
        payload = {
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": temp, "num_predict": tokens},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_S,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "") or ""
        except Exception as e:
            logger.warning("Ollama HTTP call failed (%s); retrying via SDK.", e)
            return self._chat_ollama(messages, temp, tokens, json_mode, model)

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Execute chat completion across Hugging Face or Ollama.

        Provider fallback is scoped to this call; the instance state is never
        mutated, so a judge client cannot silently degrade mid-benchmark.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        ollama_target = self.model if self.provider == "ollama" else self.ollama_model

        if self.provider == "nvidia":
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            headers = {
                "Authorization": f"Bearer {self.nvidia_key}",
                "Content-Type": "application/json",
            }
            try:
                resp = requests.post(self.nvidia_url, json=payload, headers=headers, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:
                logger.warning("NVIDIA NIM call failed (%s); falling back to Ollama.", e)

        if self.provider == "huggingface":
            for attempt in range(2):
                try:
                    res = self._hf_client.chat.completions.create(
                        model=self.model, messages=messages,
                        max_tokens=tokens, temperature=temp,
                    )
                    return res.choices[0].message.content or ""
                except Exception as e:
                    err = str(e)
                    logger.warning("HF attempt %d failed: %s", attempt + 1, e)
                    if "402" in err or "Payment Required" in err or "depleted" in err:
                        break
                    if attempt == 0:
                        time.sleep(1.0)
            logger.warning("Falling back to Ollama (%s) for this call only.", ollama_target)

        return self._chat_ollama_http(messages, temp, tokens, json_mode, ollama_target)
