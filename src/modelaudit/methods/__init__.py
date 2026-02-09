"""指纹方法注册.

导入所有方法模块以触发 @register 装饰器注册。
"""

from modelaudit.methods import llmmap, style  # noqa: F401

__all__ = ["llmmap", "style"]
