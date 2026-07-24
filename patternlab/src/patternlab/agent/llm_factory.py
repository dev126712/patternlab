"""Provider-agnostic LLM factory (same pattern as KubeAIOps).

LLM_PROVIDER env var (or derived from DEPLOYMENT_ENV: dev->groq, prod->ollama) selects the
backend. All providers must support tool calling — the agent binds detection tools.

For prod/EC2 with Ollama: use a tool-calling-capable model (qwen2.5:14b or
llama3.1:8b-instruct). Smaller models will silently fail to call tools; benchmark first.
"""
from __future__ import annotations

from patternlab.config import settings


def get_llm(temperature: float = 0.2):
    provider = settings.effective_llm_provider

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=settings.anthropic_model, temperature=temperature,
                             api_key=settings.anthropic_api_key, max_tokens=4096)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=settings.groq_model, temperature=temperature,
                        api_key=settings.groq_api_key)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=settings.ollama_model, temperature=temperature,
                          base_url=settings.ollama_base_url)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature,
                          api_key=settings.openai_api_key)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
