"""方法注册表 — 装饰器工厂模式."""


from modelaudit.base import Fingerprinter

_REGISTRY: dict[str, type[Fingerprinter]] = {}


def register(name: str):
    """注册指纹方法的装饰器.

    用法:
        @register("llmmap")
        class LLMmapFingerprinter(BlackBoxFingerprinter):
            ...
    """

    def decorator(cls: type[Fingerprinter]):
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_fingerprinter(name: str, **kwargs) -> Fingerprinter:
    """通过名称获取指纹方法实例."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"未知指纹方法: {name}。可用方法: {available}")
    return _REGISTRY[name](**kwargs)


def list_methods() -> dict[str, str]:
    """列出所有已注册方法. 返回 {name: type}."""
    return {name: cls().fingerprint_type for name, cls in sorted(_REGISTRY.items())}
