import os
import csv
import math
import random
import argparse
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


# ======================== 数据集定义 ========================

class PairDataset(Dataset):
    """
    读取训练数据，每一行是 query \t positive
    转换成 Sentence-Transformers 的 InputExample
    """
    def __init__(self, tsv_path: str):
        self.examples = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                q, p = row[0].strip(), row[1].strip()
                if q and p and q != p:
                    self.examples.append(InputExample(texts=[q, p]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_valid_pairs(tsv_path: str) -> List[Tuple[str, str]]:
    """
    加载验证集 (query, positive)
    """
    pairs = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                q, p = row[0].strip(), row[1].strip()
                if q and p and q != p:
                    pairs.append((q, p))
    return pairs


# ======================== LoRA 工具函数 ========================

def pick_lora_targets(hf_model, candidates=None):
    """
    自动选择模型里可以加 LoRA 的模块名
    不同模型的命名可能不一样，这里做一个自动探测
    """
    if candidates is None:
        candidates = [
            "query", "key", "value", "dense",
            "q_proj", "k_proj", "v_proj", "o_proj",
            "out_proj", "intermediate.dense", "output.dense"
        ]
    present = set()
    names = list(dict(hf_model.named_modules()).keys()) + list(dict(hf_model.named_parameters()).keys())
    for cand in candidates:
        for name in names:
            if f".{cand}" in name or name.endswith(cand):
                present.add(cand)
                break
    if not present:
        present = {"query", "key", "value", "dense"}
    return sorted(present)


def attach_lora_to_sbert(st_model: SentenceTransformer, r=16, alpha=32, dropout=0.05):
    """
    给 SentenceTransformer 模型挂上 LoRA 适配器
    """
    transformer: models.Transformer = st_model._first_module()
    hf_model = transformer.auto_model

    target_modules = pick_lora_targets(hf_model)
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # 用于特征提取/embedding
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        inference_mode=False,
    )
    peft_model = get_peft_model(hf_model, lora_cfg)

    transformer.auto_model = peft_model
    st_model._modules[st_model._get_module_name(0)] = transformer

    # 打印可训练参数量
    trainable, total = 0, 0
    for _, p in st_model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"[LoRA] 注入模块: {target_modules}")
    print(f"[LoRA] 可训练参数: {trainable:,} / {total:,} ({trainable/total:.2%})")

    return st_model, peft_model


# ======================== 验证函数 ========================

@torch.no_grad()
def quick_eval_top1(st_model: SentenceTransformer,
                    valid_pairs: List[Tuple[str, str]],
                    negatives_per_query: int = 9,
                    max_examples: int = 1000,
                    max_seq_len: int = 512):
    """
    简单验证：
    给每个 query，取 1 个正样 + K 个随机负样
    看 Top-1 排序准确率
    """
    if not valid_pairs:
        return 0.0

    st_model._first_module().max_seq_length = max_seq_len
    N = min(max_examples, len(valid_pairs))
    sample_pairs = random.sample(valid_pairs, N)
    correct = 0

    for q, pos in sample_pairs:
        # 随机采样负例
        negs = []
        while len(negs) < negatives_per_query:
            _, p2 = random.choice(valid_pairs)
            if p2 != pos:
                negs.append(p2)

        cands = [pos] + negs
        embs = st_model.encode([q] + cands, convert_to_tensor=True, normalize_embeddings=True)
        qv, dvs = embs[0], embs[1:]
        sims = torch.nn.functional.cosine_similarity(qv.unsqueeze(0), dvs)
        pred = int(torch.argmax(sims))
        if pred == 0:
            correct += 1

    acc = correct / N
    print(f"[验证] Top-1 准确率: {acc:.4f} (样本数={N})")
    return acc


# ======================== 主训练流程 ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="本地 m3e-large 路径 或 HuggingFace 模型名 (如 moka-ai/m3e-large)")
    parser.add_argument("--train", type=str, required=True, help="训练文件路径 (pairs.tsv)")
    parser.add_argument("--valid", type=str, default=None, help="验证文件路径 (valid_pairs.tsv)")
    parser.add_argument("--out", type=str, default="outputs/m3e-lora", help="输出目录")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-adapters", action="store_true", help="是否单独保存 LoRA adapter")
    parser.add_argument("--merge-after-train", action="store_true", help="是否合并 LoRA 到基座并导出")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 使用设备: {device}")

    # 加载基座模型（支持本地路径）
    st_model = SentenceTransformer(args.model, device=device)
    transformer = st_model._first_module()
    transformer.max_seq_length = args.max_seq_len

    # 挂载 LoRA
    st_model, peft_model = attach_lora_to_sbert(
        st_model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )

    # 加载训练数据
    train_ds = PairDataset(args.train)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, drop_last=True)

    # 加载验证数据
    valid_pairs = load_valid_pairs(args.valid) if args.valid else []

    # 定义损失
    train_loss = losses.MultipleNegativesRankingLoss(st_model)

    # 计算 warmup 步数
    total_steps = math.ceil(len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"[INFO] 总步数={total_steps}, warmup步数={warmup_steps}")

    # 按 epoch 训练
    for ep in range(1, args.epochs + 1):
        print(f"\n[训练] 第 {ep}/{args.epochs} 轮")
        st_model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=1,
            optimizer_params={"lr": args.lr},
            warmup_steps=warmup_steps if ep == 1 else 0,
            show_progress_bar=True,
            use_amp=torch.cuda.is_available(),
            output_path=args.out,
            checkpoint_path=None
        )
        if valid_pairs:
            quick_eval_top1(st_model, valid_pairs, negatives_per_query=9, max_examples=1000,
                            max_seq_len=args.max_seq_len)

    # 可选：单独保存 LoRA adapter
    if args.save_adapters:
        adapter_dir = os.path.join(args.out, "lora_adapters")
        peft_model.save_pretrained(adapter_dir)
        print(f"[保存] LoRA adapters 已保存到: {adapter_dir}")

    # 可选：合并 LoRA 权重
    if args.merge_after_train:
        print("[合并] 开始合并 LoRA 到基座模型...")
        st = SentenceTransformer(args.out, device=device)
        trans: models.Transformer = st._first_module()
        base = trans.auto_model
        base = PeftModel.from_pretrained(base, os.path.join(args.out, "lora_adapters"))
        merged = base.merge_and_unload()
        trans.auto_model = merged
        st._modules[st._get_module_name(0)] = trans
        merged_dir = os.path.join(args.out, "merged_model")
        st.save(merged_dir)
        print(f"[合并] 完整模型已保存到: {merged_dir}")

    print("\n[完成] 训练结束！")


if __name__ == "__main__":
    main()
