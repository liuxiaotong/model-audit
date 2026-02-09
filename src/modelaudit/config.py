"""配置模型."""

from typing import Literal

from pydantic import BaseModel, Field


class AuditConfig(BaseModel):
    """审计引擎配置."""

    # 黑盒配置
    blackbox_method: str = "llmmap"
    provider: Literal["openai", "anthropic", "custom"] = "openai"
    api_key: str = ""
    api_base: str = ""
    num_probes: int = 20

    # API 调用配置
    api_timeout: int = Field(default=60, ge=10, le=300, description="API 调用超时（秒）")
    api_max_retries: int = Field(default=3, ge=0, le=10, description="API 最大重试次数")

    # 白盒配置
    whitebox_method: str = "reef"
    device: str = "cpu"

    # 判定配置
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    output_format: Literal["json", "markdown"] = "markdown"
    cache_dir: str = ".modelaudit_cache"
    cache_ttl: int = Field(default=0, ge=0, description="缓存过期时间（秒），0=永不过期")
