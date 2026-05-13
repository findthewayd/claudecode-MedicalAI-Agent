"""Multi-model provider factory.

Supports:
- DeepSeek (OpenAI-compatible): deepseek-chat
- Zhipu AI / GLM (OpenAI-compatible): glm-4-plus
- Xunfei Spark: spark-lite, spark-pro, spark-max
- OpenAI (native): gpt-4, gpt-4o, etc.

Configure via .env:
  LLM_PROVIDER=deepseek|zhipu|spark|openai
  LLM_MODEL=deepseek-chat
  LLM_API_KEY=sk-xxx
  LLM_API_BASE=https://api.deepseek.com
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_llm_config():
    """Get LLM configuration from environment."""
    provider = os.environ.get("LLM_PROVIDER", "deepseek")

    configs = {
        "deepseek": {
            "api_key": os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
            "api_base": os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com"),
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        },
        "zhipu": {
            "api_key": os.environ.get("ZHIPU_API_KEY"),
            "api_base": "https://open.bigmodel.cn/api/paas/v4",
            "model": os.environ.get("ZHIPU_MODEL", "glm-4-plus"),
        },
        "spark": {
            "api_key": os.environ.get("SPARK_API_KEY"),
            "api_secret": os.environ.get("SPARK_API_SECRET"),
            "app_id": os.environ.get("SPARK_APP_ID"),
            "api_base": os.environ.get("SPARK_API_BASE", "https://spark-api-open.xf-yun.com/v1"),
            "model": os.environ.get("SPARK_MODEL", "spark-lite"),
        },
        "openai": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "api_base": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
        },
    }

    if provider not in configs:
        logger.warning(f"Unknown provider '{provider}', falling back to deepseek")
        provider = "deepseek"

    cfg = configs[provider]
    return {
        "provider": provider,
        "api_key": cfg["api_key"],
        "api_base": cfg["api_base"],
        "model": cfg.get("model", "deepseek-chat"),
        "extra": {k: v for k, v in cfg.items() if k not in ("api_key", "api_base", "model")},
    }


def create_chat_model(temperature: float = 0.2, **kwargs):
    """Create a LangChain ChatOpenAI-compatible model instance."""
    cfg = get_llm_config()
    if not cfg["api_key"]:
        raise ValueError(f"API key not configured for provider '{cfg['provider']}'")

    provider = cfg["provider"]

    if provider == "spark":
        # Spark uses its own SDK
        return _create_spark_model(cfg, temperature, **kwargs)
    else:
        # DeepSeek, Zhipu, OpenAI all use OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model_name=cfg["model"],
            temperature=temperature,
            openai_api_key=cfg["api_key"],
            openai_api_base=cfg["api_base"],
            **kwargs
        )


def _create_spark_model(cfg, temperature, **kwargs):
    """Create Xunfei Spark model via OpenAI-compatible API."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model_name=cfg["model"],
        temperature=temperature,
        openai_api_key=cfg.get("api_key", ""),
        openai_api_base=cfg["api_base"],
        default_headers={"Authorization": f"Bearer {cfg.get('api_key', '')}"},
        **kwargs
    )


# Singleton
def get_model_config_for_display():
    """Return model info for frontend display."""
    cfg = get_llm_config()
    return {
        "provider": cfg["provider"],
        "model": cfg["model"],
        "api_base": cfg["api_base"][:40] + "..." if len(cfg["api_base"]) > 40 else cfg["api_base"],
    }
