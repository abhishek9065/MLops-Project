from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import requests

from app.schemas import Citation, TokenUsage


OPENAI_INPUT_COST_PER_1M = 0.15
OPENAI_OUTPUT_COST_PER_1M = 0.60
GEMINI_INPUT_COST_PER_1M = 0.10
GEMINI_OUTPUT_COST_PER_1M = 0.40


@dataclass(frozen=True)
class LLMResult:
    answer: str
    model: str
    token_usage: TokenUsage
    estimated_cost_usd: float
    latency_ms: float


class LLMClient:
    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "local").strip().lower()
        self.model = os.getenv("LLM_MODEL", self._default_model())

    def generate(self, prompt: str, question: str, citations: list[Citation]) -> LLMResult:
        if self.provider == "openai" and os.getenv("OPENAI_API_KEY"):
            return self._generate_openai(prompt)
        if self.provider == "gemini" and os.getenv("GEMINI_API_KEY"):
            return self._generate_gemini(prompt)
        return self._generate_local(prompt, question, citations)

    def _default_model(self) -> str:
        if self.provider == "openai":
            return "gpt-4o-mini"
        if self.provider == "gemini":
            return "gemini-1.5-flash"
        return "local-extractive-rag-v1"

    def _generate_openai(self, prompt: str) -> LLMResult:
        started_at = time.perf_counter()
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {})
        token_usage = TokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
        )
        answer = payload["choices"][0]["message"]["content"]
        return LLMResult(
            answer=answer,
            model=self.model,
            token_usage=token_usage,
            estimated_cost_usd=estimate_cost(
                token_usage,
                input_cost_per_1m=OPENAI_INPUT_COST_PER_1M,
                output_cost_per_1m=OPENAI_OUTPUT_COST_PER_1M,
            ),
            latency_ms=elapsed_ms(started_at),
        )

    def _generate_gemini(self, prompt: str) -> LLMResult:
        started_at = time.perf_counter()
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            params={"key": os.environ["GEMINI_API_KEY"]},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usageMetadata", {})
        prompt_tokens = int(usage.get("promptTokenCount", 0))
        completion_tokens = int(usage.get("candidatesTokenCount", 0))
        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        answer = payload["candidates"][0]["content"]["parts"][0]["text"]
        return LLMResult(
            answer=answer,
            model=self.model,
            token_usage=token_usage,
            estimated_cost_usd=estimate_cost(
                token_usage,
                input_cost_per_1m=GEMINI_INPUT_COST_PER_1M,
                output_cost_per_1m=GEMINI_OUTPUT_COST_PER_1M,
            ),
            latency_ms=elapsed_ms(started_at),
        )

    def _generate_local(self, prompt: str, question: str, citations: list[Citation]) -> LLMResult:
        started_at = time.perf_counter()
        question_terms = normalized_terms(question)
        selected_sentences: list[str] = []

        for citation in citations:
            for sentence in split_sentences(citation.text):
                sentence_terms = normalized_terms(sentence)
                if question_terms & sentence_terms:
                    selected_sentences.append(
                        f"{sentence.strip()} [{citation.filename} chunk {citation.chunk_index}]"
                    )
                if len(selected_sentences) >= 4:
                    break
            if len(selected_sentences) >= 4:
                break

        if selected_sentences:
            answer = " ".join(selected_sentences)
        elif citations:
            first = citations[0]
            answer = (
                "I found related context, but the local fallback model could not form a precise answer. "
                f"Review [{first.filename} chunk {first.chunk_index}]."
            )
        else:
            answer = "I do not know based on the uploaded documents."

        prompt_tokens = approximate_tokens(prompt)
        completion_tokens = approximate_tokens(answer)
        return LLMResult(
            answer=answer,
            model="local-extractive-rag-v1",
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            estimated_cost_usd=0.0,
            latency_ms=elapsed_ms(started_at),
        )


def estimate_cost(token_usage: TokenUsage, input_cost_per_1m: float, output_cost_per_1m: float) -> float:
    input_cost = (token_usage.prompt_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (token_usage.completion_tokens / 1_000_000) * output_cost_per_1m
    return round(input_cost + output_cost, 8)


def approximate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def normalized_terms(text: str) -> set[str]:
    stopwords = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "are", "what", "how", "why"}
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def split_sentences(text: str) -> list[str]:
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence]


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
