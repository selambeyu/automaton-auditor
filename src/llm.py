"""
Configurable LLM layer: single provider or role-based multi-model stack.

Single-provider: LLM_PROVIDER=openai | openrouter | ollama. For vision override:
VISION_PROVIDER=gemini with GOOGLE_API_KEY.

Multi-model stack (set per-role env vars; else falls back to get_llm/get_vision_llm):
- Judicial bench: JUDICIAL_PROVIDER=groq → Groq (Llama 70B-class)
- Vision & PDF detective: DETECTIVE_PROVIDER=gemini → Gemini 2.5 Pro
- Forensic investigator: FORENSIC_PROVIDER=openai → OpenAI GPT-4o-mini
"""

from __future__ import annotations

import os
from langchain_core.language_models.chat_models import BaseChatModel


def _provider() -> str:
    """Return lowercase LLM_PROVIDER; default openai if unset."""
    return (os.environ.get("LLM_PROVIDER") or "openai").strip().lower()


def get_vision_provider() -> str:
    """
    Return the vision provider: VISION_PROVIDER if set (e.g. gemini), else LLM_PROVIDER.
    Used by vision_inspector_node to choose message format (e.g. Gemini-native image block).
    """
    return (os.environ.get("VISION_PROVIDER") or os.environ.get("LLM_PROVIDER") or "openai").strip().lower()


def get_llm() -> BaseChatModel:
    """
    Return the chat model for Judges (and any text-only nodes).
    - openai: ChatOpenAI (OPENAI_API_KEY, OPENAI_MODEL)
    - openrouter: ChatOpenAI with OpenRouter base_url (OPENROUTER_API_KEY, OPENROUTER_MODEL)
    - ollama: ChatOllama local (OLLAMA_BASE_URL, OLLAMA_MODEL)
    """
    provider = _provider()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.3,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.3,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        )

    # default: openai
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not (api_key and str(api_key).strip()):
        raise ValueError(
            "OPENAI_API_KEY is not set. Either set it in .env or use another provider: "
            "LLM_PROVIDER=openrouter (with OPENROUTER_API_KEY) or LLM_PROVIDER=ollama (no key, local)."
        )
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        api_key=api_key,
    )


def get_vision_llm() -> BaseChatModel | None:
    """
    Return a vision-capable model for VisionInspector (image input).
    - gemini: ChatGoogleGenerativeAI (VISION_PROVIDER=gemini, GOOGLE_API_KEY, GOOGLE_VISION_MODEL)
    - openai: ChatOpenAI with vision model (OPENAI_VISION_MODEL)
    - openrouter: same client, OPENROUTER_VISION_MODEL (e.g. openai/gpt-4o)
    - ollama: ChatOllama with vision model (OLLAMA_VISION_MODEL, e.g. llava)
    Returns None if vision is not configured or provider has no vision model.
    """
    provider = get_vision_provider()

    if provider == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not (api_key and str(api_key).strip()):
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            return None
        return ChatGoogleGenerativeAI(
            model=os.environ.get("GOOGLE_VISION_MODEL", "gemini-2.0-flash"),
            temperature=0.2,
            google_api_key=api_key,
        )

    provider = _provider()
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.environ.get("OLLAMA_VISION_MODEL", "llava")
        return ChatOllama(
            model=model,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.2,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        model = os.environ.get("OPENROUTER_VISION_MODEL") or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            return None
        return ChatOpenAI(
            model=model,
            temperature=0.2,
            api_key=key,
            base_url=os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        )

    from langchain_openai import ChatOpenAI

    model = os.environ.get("OPENAI_VISION_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    return ChatOpenAI(
        model=model,
        temperature=0.2,
        api_key=key,
    )


def _judicial_provider() -> str:
    """Return JUDICIAL_PROVIDER if set, else empty (use get_llm)."""
    return (os.environ.get("JUDICIAL_PROVIDER") or "").strip().lower()


def _detective_provider() -> str:
    """Return DETECTIVE_PROVIDER if set, else empty (use get_vision_llm / get_llm)."""
    return (os.environ.get("DETECTIVE_PROVIDER") or "").strip().lower()


def get_detective_provider() -> str:
    """Return DETECTIVE_PROVIDER if set (e.g. gemini). Used for message format (e.g. Gemini image block)."""
    return _detective_provider()


def _forensic_provider() -> str:
    """Return FORENSIC_PROVIDER if set, else empty (use get_llm)."""
    return (os.environ.get("FORENSIC_PROVIDER") or "").strip().lower()


def get_judicial_llm() -> BaseChatModel:
    """
    Judicial bench: Prosecutor, Defense, Tech Lead.
    When JUDICIAL_PROVIDER=groq and GROQ_API_KEY set, returns ChatGroq (e.g. Llama 70B).
    Else delegates to get_llm() for backward compatibility.
    """
    if _judicial_provider() == "groq":
        api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
        if api_key:
            try:
                from langchain_groq import ChatGroq

                return ChatGroq(
                    model=os.environ.get("GROQ_JUDICIAL_MODEL", "llama-3.3-70b-versatile"),
                    temperature=0.3,
                    groq_api_key=api_key,
                )
            except ImportError:
                pass
    return get_llm()


def get_detective_llm() -> BaseChatModel:
    """
    Vision & PDF detective: VisionInspector, DocAnalyst (theoretical depth).
    When DETECTIVE_PROVIDER=gemini and GOOGLE_API_KEY set, returns ChatGoogleGenerativeAI (e.g. Gemini 2.5 Pro).
    Else delegates to get_vision_llm() if available, else get_llm().
    """
    if _detective_provider() == "gemini":
        api_key = (os.environ.get("GOOGLE_API_KEY") or "").strip()
        if api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(
                    model=os.environ.get("GOOGLE_DETECTIVE_MODEL", "gemini-2.5-pro"),
                    temperature=0.2,
                    google_api_key=api_key,
                )
            except ImportError:
                pass
    vision = get_vision_llm()
    return vision if vision is not None else get_llm()


def get_forensic_llm() -> BaseChatModel:
    """
    Forensic investigator: optional RepoInvestigator code-validation step.
    When FORENSIC_PROVIDER=openai and OPENAI_API_KEY set, returns ChatOpenAI (e.g. GPT-4o-mini).
    Else delegates to get_llm(). Documented for future use in RepoInvestigator.
    """
    if _forensic_provider() == "openai":
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if api_key:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=os.environ.get("OPENAI_FORENSIC_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                api_key=api_key,
            )
    return get_llm()
