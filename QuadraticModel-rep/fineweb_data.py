"""
QuadraticModel/data.py 的整份副本（严格复现，不 import 原版）。

与原版逐字一致，仅一处改动：tokenizer 的查找路径从相对 "tokenizers/bpe_*.json"
改为「相对本文件所在目录」，这样从 QuadraticModel-rep 任意 cwd 启动都能命中同一个
bpe_8192.json（其 sha1 与原版一致），不会误触发 BPE 重训。其余采样逻辑
（shards 划分、seed 全局 shuffle、逐文档追加 <eot>、ConcatThenSplit 切窗、
顺序 batch）与原版 QuadraticModel/data.py 完全相同。
"""
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import grain
import numpy as np
from array_record.python.array_record_module import ArrayRecordReader
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# 仅此一处偏离原版：tokenizer 目录锚定到本文件旁，避免依赖 cwd。
_TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizers"


@dataclass(frozen=True)
class FineWeb:
    seq_len: int
    vocab_size: int
    arrayrecord_dir: str | None = None

    @cached_property
    def tokenizer(self):
        path = _TOKENIZER_DIR / f"bpe_{self.vocab_size}.json"
        if path.exists():
            print(f"Loading tokenizer from {path}")
            return Tokenizer.from_file(str(path))
        print(f"Training tokenizer with vocab_size={self.vocab_size}...")
        path.parent.mkdir(exist_ok=True)
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<eot>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        def iter_shards():
            for shard in self.shards[:2]:
                reader = ArrayRecordReader(str(shard))
                for _ in range(reader.num_records()):
                    yield reader.read().decode("utf-8", errors="ignore")
                reader.close()

        tokenizer.train_from_iterator(iter_shards(), trainer=trainer)
        tokenizer.save(str(path))
        print(f"Saved {path} ({tokenizer.get_vocab_size()} tokens)")
        return tokenizer

    @cached_property
    def eot_token(self) -> int:
        return self.tokenizer.token_to_id("<eot>")

    @cached_property
    def token_bytes(self) -> np.ndarray:
        id_to_piece = {v: k for k, v in self.tokenizer.get_vocab().items()}
        return np.array(
            [
                len(id_to_piece.get(i, ""))
                for i in range(self.tokenizer.get_vocab_size())
            ],
            dtype=np.int32,
        )

    @cached_property
    def shards(self) -> list[Path]:
        data_dir = (
            Path(self.arrayrecord_dir)
            if self.arrayrecord_dir is not None
            else Path(os.environ["DATA_DIR"]) / "fineweb_edu_100B_arrayrecord"
        )
        shards = sorted(data_dir.glob("*.arrayrecord"))
        if len(shards) < 2:
            raise ValueError(
                f"Expected at least two .arrayrecord shards under {data_dir}, "
                f"found {len(shards)}"
            )
        return shards

    def build(
        self,
        split: Literal["train", "eval"],
        batch_size: int,
        seed: int,
        shuffle: bool,
        repeat: bool,
    ) -> grain.IterDataset:
        shards = self.shards[:-1] if split == "train" else self.shards[-1:]
        tokenizer = self.tokenizer
        token_bytes = self.token_bytes
        eot = self.eot_token

        def tokenize(rec):
            text = rec.decode("utf-8", errors="ignore")
            enc = np.asarray(tokenizer.encode(text).ids, dtype=np.int32)
            tokens = np.empty(enc.size + 1, dtype=np.int32)
            tokens[:-1] = enc
            tokens[-1] = eot
            return {"tokens": tokens, "bytes": token_bytes[tokens]}

        source = grain.sources.ArrayRecordDataSource(shards)
        ds = grain.MapDataset.source(source).seed(seed)
        if shuffle:
            ds = ds.shuffle()
        if repeat:
            ds = ds.repeat()
        ds = ds.map(tokenize).to_iter_dataset()
        ds = grain.experimental.ConcatThenSplitIterDataset(
            ds, length_struct={"tokens": self.seq_len + 1, "bytes": self.seq_len + 1}
        )

        def split_batch(batch):
            tokens, n_bytes = batch["tokens"], batch["bytes"]
            x, y = tokens[:-1], tokens[1:]
            n_bytes = n_bytes[1:]
            return (x, y), n_bytes

        ds = ds.map(split_batch)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds


# ---------------------------------------------------------------------------
# parquet 源变体（避免 arrayrecord 转换占 208G 配额；接受与原版的一处偏离）
# ---------------------------------------------------------------------------
#
# 与原版 FineWeb.build 的唯一差异：数据源与 shuffle。
#   原版：ArrayRecordDataSource（随机访问）+ MapDataset.shuffle()（全局索引全排列）
#   这里：ParquetIterDataset（顺序流）      + WindowShuffleIterDataset（滑窗近似洗牌）
# 其余 tokenize（逐文档追加 <eot>）、ConcatThenSplit 切 (seq_len+1) 窗、split_batch、
# batch(drop_remainder) 全部与原版逐行一致。
#
# ⚠ 因 grain 对 parquet 只提供顺序 IterDataset（无 __getitem__），无法做原版那种
# 基于全局索引的 shuffle，只能用固定窗口的滑窗洗牌。故文档顺序与原版不同 → 逐 batch
# 内容不同（非 bit 级复现）。采样"语义"仍是"打乱文档→拼接→切窗"，但不保证与 JAX run
# 逐 batch 一致。这是用户在配额受限下明确接受的取舍。
SHUFFLE_WINDOW = 100_000  # 滑窗洗牌窗口（文档数）；越大越接近全局 shuffle，内存也越高


@dataclass(frozen=True)
class FineWebParquet:
    seq_len: int
    vocab_size: int
    parquet_dir: str | None = None
    shuffle_window: int = SHUFFLE_WINDOW

    # tokenizer / eot_token / token_bytes 与 FineWeb 完全相同（复用同一份 bpe_8192）。
    @cached_property
    def tokenizer(self):
        path = _TOKENIZER_DIR / f"bpe_{self.vocab_size}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"parquet 源不支持从数据训练 tokenizer；需预置 {path}"
            )
        print(f"Loading tokenizer from {path}")
        return Tokenizer.from_file(str(path))

    @cached_property
    def eot_token(self) -> int:
        return self.tokenizer.token_to_id("<eot>")

    @cached_property
    def token_bytes(self) -> np.ndarray:
        id_to_piece = {v: k for k, v in self.tokenizer.get_vocab().items()}
        return np.array(
            [len(id_to_piece.get(i, "")) for i in range(self.tokenizer.get_vocab_size())],
            dtype=np.int32,
        )

    @cached_property
    def files(self) -> list[Path]:
        data_dir = (
            Path(self.parquet_dir)
            if self.parquet_dir is not None
            else Path(os.environ["DATA_DIR"]) / "fineweb_edu_100B_parquet" / "sample" / "100BT"
        )
        files = sorted(data_dir.glob("*.parquet"))
        if len(files) < 2:
            raise ValueError(
                f"Expected at least two .parquet files under {data_dir}, found {len(files)}"
            )
        return files

    def build(
        self,
        split: Literal["train", "eval"],
        batch_size: int,
        seed: int,
        shuffle: bool,
        repeat: bool,
    ) -> grain.IterDataset:
        # train=files[:-1] / eval=files[-1:]，与原版 shards 划分一致（最后一个 shard 作 eval）。
        files = self.files[:-1] if split == "train" else self.files[-1:]
        tokenizer = self.tokenizer
        token_bytes = self.token_bytes
        eot = self.eot_token

        def tokenize(rec):
            # parquet 每条 record 是 dict，取 text（str）；原版是 bytes.decode。
            text = rec["text"]
            if text is None:
                text = ""
            enc = np.asarray(tokenizer.encode(text).ids, dtype=np.int32)
            tokens = np.empty(enc.size + 1, dtype=np.int32)
            tokens[:-1] = enc
            tokens[-1] = eot
            return {"tokens": tokens, "bytes": token_bytes[tokens]}

        ds = grain.experimental.ParquetIterDataset([str(p) for p in files])
        # ⚠ 顺序：先 repeat 再 shuffle（与原版 .shuffle().repeat() 相反，故意如此）。
        # 原版全局 shuffle 每个 epoch 重新洗牌，靠的是 MapDataset 的随机访问；这里只有
        # 滑窗洗牌。若「先 shuffle 再 repeat」，会把同一个滑窗结果每 epoch 原样重播（随机
        # 性差）。改成「先 repeat 再 shuffle」→ 无限流持续喂进滑窗缓冲，每次经过窗口的
        # 采样都不同，是滑窗方案下更接近全局 shuffle 的做法。
        if repeat:
            ds = grain.experimental.RepeatIterDataset(ds, num_epochs=None)
        if shuffle:
            ds = grain.experimental.WindowShuffleIterDataset(
                ds, window_size=self.shuffle_window, seed=seed
            )
        ds = ds.map(tokenize)
        ds = grain.experimental.ConcatThenSplitIterDataset(
            ds, length_struct={"tokens": self.seq_len + 1, "bytes": self.seq_len + 1}
        )

        def split_batch(batch):
            tokens, n_bytes = batch["tokens"], batch["bytes"]
            x, y = tokens[:-1], tokens[1:]
            n_bytes = n_bytes[1:]
            return (x, y), n_bytes

        ds = ds.map(split_batch)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds
