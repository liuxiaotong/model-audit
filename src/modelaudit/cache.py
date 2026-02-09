"""指纹缓存 — 避免重复调用 API.

将模型指纹保存为本地 JSON 文件，下次审计同一模型时直接复用。
"""

import json
from pathlib import Path

from modelaudit.models import Fingerprint


class FingerprintCache:
    """本地指纹缓存."""

    def __init__(self, cache_dir: str = ".modelaudit_cache"):
        self.cache_dir = Path(cache_dir)

    def get(self, model: str, method: str, provider: str) -> Fingerprint | None:
        """从缓存读取指纹. 不存在则返回 None."""
        path = self.cache_dir / f"{self._key(model, method, provider)}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Fingerprint(**data)
        except (json.JSONDecodeError, Exception):
            return None

    def put(self, model: str, method: str, provider: str, fp: Fingerprint) -> None:
        """将指纹写入缓存."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{self._key(model, method, provider)}.json"
        path.write_text(
            json.dumps(fp.model_dump(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def list_entries(self) -> list[dict[str, str]]:
        """列出所有缓存条目."""
        entries = []
        if not self.cache_dir.exists():
            return entries

        for path in sorted(self.cache_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entries.append({
                    "file": path.name,
                    "model": data.get("model_id", ""),
                    "method": data.get("method", ""),
                    "type": data.get("fingerprint_type", ""),
                    "created": data.get("created_at", ""),
                    "size": f"{path.stat().st_size / 1024:.1f} KB",
                })
            except (json.JSONDecodeError, Exception):
                entries.append({
                    "file": path.name,
                    "model": "?",
                    "method": "?",
                    "type": "?",
                    "created": "?",
                    "size": f"{path.stat().st_size / 1024:.1f} KB",
                })

        return entries

    def clear(self) -> int:
        """清除所有缓存. 返回删除的文件数."""
        count = 0
        if self.cache_dir.exists():
            for path in self.cache_dir.glob("*.json"):
                path.unlink()
                count += 1
        return count

    @staticmethod
    def _key(model: str, method: str, provider: str) -> str:
        """生成缓存文件名."""
        safe_model = model.replace("/", "_").replace(":", "_").replace(" ", "_")
        return f"{method}_{safe_model}_{provider}"
