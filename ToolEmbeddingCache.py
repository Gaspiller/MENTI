import os
import json
import hashlib
import numpy as np
from typing import Dict, List, Tuple


class ToolEmbeddingCache:
    """
    磁盘级工具向量缓存：
    - 针对工具清单（function_name/tool_name/docstring/description）构建三种视图并编码
    - 通过(工具字段内容哈希 + 模型名)作为缓存键，自动失效重算
    - 返回形如 {"v1": np.ndarray, "v2": np.ndarray, "v3": np.ndarray}
    """

    def __init__(self, embedding_model, cache_dir: str = "./cache", normalize: bool = True) -> None:
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.normalize = normalize
        os.makedirs(self.cache_dir, exist_ok=True)

    def _file_hash(self, tools: List[dict]) -> str:
        payload = [
            {
                "function_name": t.get("function_name", ""),
                "tool_name": t.get("tool_name", ""),
                "docstring": t.get("docstring", ""),
                "description": t.get("description", ""),
            }
            for t in tools
        ]
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _model_name(self) -> str:
        name = getattr(self.embedding_model, "name_or_path", None)
        if not name:
            name = getattr(self.embedding_model, "__class__", type("X", (), {})).__name__
        return hashlib.md5(str(name).encode("utf-8")).hexdigest()[:10]

    def _cache_path(self, toolkit: str, tools: List[dict]) -> str:
        fid = self._file_hash(tools)
        mid = self._model_name()
        filename = f"embed_{toolkit}_{fid}_{mid}.npz"
        return os.path.join(self.cache_dir, filename)

    def _build_views(self, tools: List[dict]) -> Tuple[List[str], List[str], List[str]]:
        v1 = [f'{t.get("function_name", "")}\n\n{t.get("tool_name", "")}' for t in tools]
        v2 = [
            f'{t.get("function_name", "")}\n\n{t.get("tool_name", "")}\n\n{t.get("docstring", "")}'
            for t in tools
        ]
        v3 = [
            f'{t.get("function_name", "")}\n\n{t.get("tool_name", "")}\n\n{t.get("description", "")}'
            for t in tools
        ]
        return v1, v2, v3

    def _l2_normalize(self, vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return vectors / norms

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        # 兼容不同 embedding 库返回类型
        embs = self.embedding_model.encode(texts)
        embs = np.asarray(embs)
        if self.normalize:
            embs = self._l2_normalize(embs)
        return embs

    def get_embeddings(self, toolkit: str, tools: List[dict]) -> Dict[str, np.ndarray]:
        path = self._cache_path(toolkit, tools)
        if os.path.exists(path):
            data = np.load(path)
            return {"v1": data["v1"], "v2": data["v2"], "v3": data["v3"]}

        v1_txt, v2_txt, v3_txt = self._build_views(tools)
        v1 = self._encode_batch(v1_txt)
        v2 = self._encode_batch(v2_txt)
        v3 = self._encode_batch(v3_txt)

        np.savez(path, v1=v1, v2=v2, v3=v3)
        return {"v1": v1, "v2": v2, "v3": v3}


