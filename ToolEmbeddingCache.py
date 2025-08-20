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
        # 结合底层路径和嵌入维度，确保不同模型/维度不共用缓存
        dim = None
        try:
            dim = getattr(self.embedding_model, "get_sentence_embedding_dimension", lambda: None)()
        except Exception:
            dim = None

        path = None
        try:
            fm = getattr(self.embedding_model, "_first_module", None)
            if callable(fm):
                fm = fm()
            if fm is not None:
                path = (
                    getattr(getattr(fm, "auto_model", None), "name_or_path", None)
                    or getattr(getattr(fm, "model", None), "name_or_path", None)
                    or getattr(fm, "name_or_path", None)
                )
        except Exception:
            path = None

        if path is None:
            try:
                mcd = getattr(self.embedding_model, "model_card_data", None)
                if isinstance(mcd, dict):
                    path = mcd.get("model_id", None)
            except Exception:
                pass

        base = f"{path or ''}|dim={dim or ''}|{self.embedding_model.__class__.__name__}"
        return hashlib.md5(base.encode("utf-8")).hexdigest()[:10]

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

        def _recompute_and_save() -> Dict[str, np.ndarray]:
            v1_txt, v2_txt, v3_txt = self._build_views(tools)
            v1 = self._encode_batch(v1_txt)
            v2 = self._encode_batch(v2_txt)
            v3 = self._encode_batch(v3_txt)
            np.savez(path, v1=v1, v2=v2, v3=v3)
            return {"v1": v1, "v2": v2, "v3": v3}

        if os.path.exists(path):
            try:
                data = np.load(path)
                v1, v2, v3 = data["v1"], data["v2"], data["v3"]
                # 维度校验，不匹配则重算
                try:
                    dim = getattr(self.embedding_model, "get_sentence_embedding_dimension", lambda: None)()
                except Exception:
                    dim = None
                if dim and ((v1.ndim == 2 and v1.shape[1] != dim) or (v2.ndim == 2 and v2.shape[1] != dim) or (v3.ndim == 2 and v3.shape[1] != dim)):
                    return _recompute_and_save()
                return {"v1": v1, "v2": v2, "v3": v3}
            except Exception:
                # 旧格式或损坏，直接重算
                return _recompute_and_save()

        return _recompute_and_save()


