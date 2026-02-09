"""指纹方法抽象基类."""

from abc import ABC, abstractmethod
from typing import Literal

from modelaudit.models import ComparisonResult, Fingerprint


class Fingerprinter(ABC):
    """指纹方法抽象基类.

    所有指纹方法（白盒/黑盒）都需要实现这三个方法:
    - prepare(): 加载模型或建立 API 连接
    - get_fingerprint(): 提取模型指纹
    - compare(): 比对两个指纹
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """方法名称."""

    @property
    @abstractmethod
    def fingerprint_type(self) -> Literal["whitebox", "blackbox"]:
        """指纹类型."""

    @abstractmethod
    def prepare(self, model: str, **kwargs) -> None:
        """准备阶段: 加载模型权重（白盒）或建立 API 连接（黑盒）."""

    @abstractmethod
    def get_fingerprint(self) -> Fingerprint:
        """提取模型指纹."""

    @abstractmethod
    def compare(self, fp_a: Fingerprint, fp_b: Fingerprint) -> ComparisonResult:
        """比对两个指纹."""


class WhiteBoxFingerprinter(Fingerprinter):
    """白盒指纹基类，需要访问模型权重."""

    @property
    def fingerprint_type(self) -> Literal["whitebox", "blackbox"]:
        return "whitebox"


class BlackBoxFingerprinter(Fingerprinter):
    """黑盒指纹基类，只需要 API 访问."""

    @property
    def fingerprint_type(self) -> Literal["whitebox", "blackbox"]:
        return "blackbox"
