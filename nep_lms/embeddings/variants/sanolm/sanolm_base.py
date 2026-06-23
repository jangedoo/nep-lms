"""Pretraining recipe for the 50M parameter SANOLM encoder.

SANOLM is a masked-language model, not a SentenceTransformer training job, so it
intentionally has its own small training abstraction.  The resulting
``ModernBertForMaskedLM`` checkpoint can later be used as the encoder for a
sentence embedding model.

Examples
--------
Run a cheap smoke test (the ``answer`` column from Nepali QA 9k)::

    poetry run python nep_lms/embeddings/variants/sanolm/sanolm_base.py \
        --quick-run --max-steps 100 --no-push-to-hub

Run the full streaming corpus on a CUDA machine::

    poetry run python nep_lms/embeddings/variants/sanolm/sanolm_base.py \
        --max-steps 250000 --output-dir outputs/sanolm-v1-base
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    ModernBertConfig,
    ModernBertForMaskedLM,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 100
TOKENIZER_ID = "jangedoo/sanolm-tokenizer"
HUB_MODEL_ID = "jangedoo/sanolm-v1-base"
REDDIT_CONFIGS = ("comments", "posts")
REDDIT_SPLITS = ("nepalsocial", "nepalstock", "technepal")


@dataclass(frozen=True)
class HardwareSettings:
    """Training settings derived from the machine running this process."""

    device: str
    gpu_name: str | None
    gpu_vram_gib: float | None
    system_ram_gib: float
    cpu_count: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    dataloader_num_workers: int
    sequence_length: int
    use_bf16: bool
    use_fp16: bool
    use_tf32: bool
    gradient_checkpointing: bool

    @classmethod
    def detect(
        cls, effective_batch_size: int = 256, sequence_length: int = 1_024
    ) -> "HardwareSettings":
        """Choose conservative high-throughput defaults without hard-coding a GPU.

        ``auto_find_batch_size`` remains enabled in the training arguments.  The
        estimate below therefore starts large on capable GPUs and automatically
        halves if a different model/driver has a lower usable memory ceiling.
        The estimate is scaled from a 512-token reference batch, so changing the
        model context length also changes the initial batch estimate.
        """

        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        cpu_count = os.cpu_count() or 1
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
            ram_gib = page_size * page_count / 1024**3
        except (AttributeError, OSError, ValueError):
            ram_gib = 0.0

        if not torch.cuda.is_available():
            # CPU is intended only for smoke tests.  Keeping a small batch avoids
            # swapping while still preserving the same effective batch size.
            batch_size = 4 if ram_gib >= 16 else 2
            return cls(
                device="cpu",
                gpu_name=None,
                gpu_vram_gib=None,
                system_ram_gib=ram_gib,
                cpu_count=cpu_count,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=math.ceil(effective_batch_size / batch_size),
                # A CPU run is the smoke-test path.  Keeping it single-process
                # makes it reliable in notebooks and constrained containers.
                dataloader_num_workers=0,
                sequence_length=sequence_length,
                use_bf16=False,
                use_fp16=False,
                use_tf32=False,
                gradient_checkpointing=True,
            )

        properties = torch.cuda.get_device_properties(0)
        vram_gib = properties.total_memory / 1024**3
        # Reference batches are for this 49M-parameter model at 512 tokens. A
        # 24GB RTX Pro GPU at 1,024 tokens therefore starts at batch 32.
        if vram_gib >= 20:
            reference_batch_size = 64
        elif vram_gib >= 16:
            reference_batch_size = 48
        elif vram_gib >= 12:
            reference_batch_size = 32
        elif vram_gib >= 8:
            reference_batch_size = 16
        else:
            reference_batch_size = 8
        scaled_batch_size = max(1, int(reference_batch_size * 512 / sequence_length))
        # Power-of-two batches are efficient on Tensor Cores and make a stable
        # starting point for Trainer's automatic OOM batch-size backoff.
        batch_size = 2 ** int(math.floor(math.log2(scaled_batch_size)))

        bf16 = torch.cuda.is_bf16_supported()
        # Streaming/token packing benefits from parallel workers, but excessive
        # workers compete for RAM with the Arrow and tokenizer buffers.
        workers_from_ram = max(2, int(ram_gib // 4))
        return cls(
            device="cuda",
            gpu_name=properties.name,
            gpu_vram_gib=round(vram_gib, 1),
            system_ram_gib=ram_gib,
            cpu_count=cpu_count,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=math.ceil(effective_batch_size / batch_size),
            dataloader_num_workers=min(8, max(2, cpu_count - 2), workers_from_ram),
            sequence_length=sequence_length,
            use_bf16=bf16,
            use_fp16=not bf16,
            use_tf32=True,
            # Checkpointing is useful below 20GB; disabling it on larger cards is
            # materially faster and the OOM backoff still protects the run.
            gradient_checkpointing=vram_gib < 20,
        )


class MLMTrainer(Trainer):
    """Trainer that logs perplexity in the same event as validation loss."""

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "eval_loss" in logs:
            # Guarding the exponent prevents an unusable initial checkpoint from
            # turning the metrics JSON into infinity.
            logs["eval_perplexity"] = float(math.exp(min(logs["eval_loss"], 20)))
        super().log(logs, start_time)


class SANOLMBase:
    """A resource-adaptive ModernBERT masked-LM pretraining recipe.

    The config is deliberately compact: 8 layers, 512 hidden dimensions, 8
    heads, and a 1,536-dimension MLP.  With SANOLM's 42k-token vocabulary this
    is 49.1M trainable parameters, close to the requested 50M target.
    """

    hub_model_id = HUB_MODEL_ID
    tokenizer_id = TOKENIZER_ID
    # Train on 1k contexts now; retain an 8k RoPE configuration for later
    # continued pretraining or long-context embedding fine-tuning.
    max_sequence_length = 1_024
    max_document_tokens = 4_096
    max_position_embeddings = 8_192
    effective_batch_size = 256
    compile_model = True

    def __init__(
        self,
        *,
        output_dir: str | Path = "outputs/sanolm-v1-base",
        tokenizer_id: str = TOKENIZER_ID,
        hub_model_id: str = HUB_MODEL_ID,
        seed: int = SEED,
        compile_model: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.tokenizer_id = tokenizer_id
        self.hub_model_id = hub_model_id
        self.seed = seed
        self.compile_model = compile_model
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: ModernBertForMaskedLM | None = None
        self._hardware: HardwareSettings | None = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id,
                model_max_length=self.max_sequence_length,
                use_fast=True,
            )
            if self._tokenizer.pad_token_id is None:
                raise ValueError("SANOLM tokenizer must define a pad token")
        return self._tokenizer

    @property
    def hardware(self) -> HardwareSettings:
        if self._hardware is None:
            self._hardware = HardwareSettings.detect(
                self.effective_batch_size, self.max_sequence_length
            )
        return self._hardware

    def get_model(self) -> ModernBertForMaskedLM:
        """Create a new 49.1M parameter ModernBERT MLM model."""

        tokenizer = self.tokenizer
        config = ModernBertConfig(
            vocab_size=len(tokenizer),
            hidden_size=512,
            intermediate_size=1_536,
            num_hidden_layers=8,
            num_attention_heads=8,
            max_position_embeddings=self.max_position_embeddings,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            cls_token_id=tokenizer.bos_token_id,
            # This keeps local attention efficient while retaining ModernBERT's
            # periodic global layers and 8k RoPE configuration.
            local_attention=128,
            attention_dropout=0.0,
            embedding_dropout=0.0,
            mlp_dropout=0.0,
            tie_word_embeddings=True,
            use_cache=False,
        )
        model = ModernBertForMaskedLM(config)
        model.config._name_or_path = self.hub_model_id
        if self.compile_model:
            # Use PyTorch's current precision API.  Do not set Trainer's legacy
            # ``tf32`` option here: Inductor rejects a mixed legacy/new TF32
            # state, while this enables Tensor Cores for any FP32 matmuls left
            # around the BF16 autocast path.
            if torch.cuda.is_available():
                torch.set_float32_matmul_precision("high")
            model = torch.compile(model)
        return model

    @property
    def model(self) -> ModernBertForMaskedLM:
        if self._model is None:
            self._model = self.get_model()
        return self._model

    @staticmethod
    def _stable_eval_bucket(text: str) -> bool:
        """Reserve a deterministic 2% holdout without materialising the corpus."""

        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % 100 < 2

    @staticmethod
    def _clean_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple)):
            return "\n".join(str(item).strip() for item in value if str(item).strip())
        return ""

    def _reddit_row_to_text(self, row: dict[str, Any]) -> dict[str, str]:
        # The comments config uses ``body``.  Posts have historically used title
        # plus body/selftext, so accept both schema variants and avoid duplicating
        # the same string if a future dataset revision exposes aliases.
        parts: list[str] = []
        seen: set[str] = set()
        for key in ("title", "selftext", "body", "text", "content"):
            text = self._clean_text(row.get(key))
            if text and text not in seen and text not in {"[deleted]", "[removed]"}:
                parts.append(text)
                seen.add(text)
        return {"text": "\n".join(parts)}

    def _text_stream(
        self, dataset_id: str, *, name: str | None = None, split: str = "train"
    ) -> datasets.IterableDataset:
        return datasets.load_dataset(dataset_id, name=name, split=split, streaming=True)

    def _full_text_streams(self) -> list[datasets.IterableDataset]:
        """Load English, Devanagari Nepali, and every requested Reddit partition."""

        nepali = self._text_stream("jangedoo/nepali-corpus").select_columns(["text"])
        english = self._text_stream(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT"
        ).select_columns(["text"])
        streams = [nepali, english]
        for config in REDDIT_CONFIGS:
            for split in REDDIT_SPLITS:
                reddit = self._text_stream(
                    "jangedoo/nepali-reddit", name=config, split=split
                )
                # ``map`` preserves streaming and keeps this tolerant of the two
                # different Reddit schemas.
                reddit = reddit.map(self._reddit_row_to_text)
                streams.append(reddit.select_columns(["text"]))
        return streams

    def _split_stream(
        self, stream: datasets.IterableDataset, *, evaluation: bool
    ) -> datasets.IterableDataset:
        def belongs_to_split(row: dict[str, Any]) -> bool:
            text = self._clean_text(row.get("text"))
            return bool(text) and self._stable_eval_bucket(text) == evaluation

        return stream.filter(belongs_to_split)

    def _quick_text_datasets(
        self, max_examples: int
    ) -> tuple[datasets.Dataset, datasets.Dataset]:
        """Return an answer-only Nepali QA 9k split for a fast integration run."""

        qa = datasets.load_dataset("jangedoo/nepali-qa-9k")
        if isinstance(qa, datasets.DatasetDict):
            qa = datasets.concatenate_datasets(list(qa.values()))
        if "answer" not in qa.column_names:
            raise ValueError("jangedoo/nepali-qa-9k must have an 'answer' column")
        text = qa.select_columns(["answer"]).rename_column("answer", "text")
        text = text.filter(lambda row: bool(self._clean_text(row["text"])))
        text = text.shuffle(seed=self.seed)
        text = text.select(range(min(max_examples, len(text))))
        split = text.train_test_split(test_size=0.1, seed=self.seed)
        return split["train"], split["test"]

    def get_train_eval_text_datasets(
        self, *, quick_run: bool = False, quick_examples: int = 4_000, eval_examples: int = 20_000
    ) -> tuple[datasets.Dataset | datasets.IterableDataset, datasets.Dataset | datasets.IterableDataset]:
        """Build deterministic train/test text datasets before token packing."""

        if quick_run:
            return self._quick_text_datasets(quick_examples)

        streams = self._full_text_streams()
        train_streams = [self._split_stream(stream, evaluation=False) for stream in streams]
        eval_streams = [self._split_stream(stream, evaluation=True) for stream in streams]
        # Preserve the tokenizer's desired language mix.  Reddit's 20% share is
        # split evenly over comments/posts and all three subreddit splits.
        probabilities = [0.60, 0.20] + [0.20 / len(REDDIT_CONFIGS) / len(REDDIT_SPLITS)] * 6
        train = datasets.interleave_datasets(
            train_streams,
            probabilities=probabilities,
            seed=self.seed,
            stopping_strategy="all_exhausted",
        ).shuffle(seed=self.seed, buffer_size=20_000)
        evaluation = datasets.interleave_datasets(
            eval_streams,
            probabilities=probabilities,
            seed=self.seed + 1,
            stopping_strategy="all_exhausted",
        ).take(eval_examples)
        return train, evaluation

    def _tokenize_and_pack(
        self, dataset: datasets.Dataset | datasets.IterableDataset
    ) -> datasets.Dataset | datasets.IterableDataset:
        """Pack documents into fixed 1,024-token blocks to minimise padding work."""

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("SANOLM tokenizer must define an EOS token")

        def tokenize_and_pack(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
            texts = [self._clean_text(text) for text in batch["text"]]
            texts = [text for text in texts if text]
            if not texts:
                return {"input_ids": [], "attention_mask": []}
            encoded = self.tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_document_tokens,
                return_attention_mask=False,
            )
            tokens: list[int] = []
            for input_ids in encoded["input_ids"]:
                tokens.extend(input_ids)
                tokens.append(eos_token_id)
            usable_length = len(tokens) - len(tokens) % self.max_sequence_length
            input_ids = [
                tokens[index : index + self.max_sequence_length]
                for index in range(0, usable_length, self.max_sequence_length)
            ]
            return {
                "input_ids": input_ids,
                "attention_mask": [[1] * self.max_sequence_length for _ in input_ids],
            }

        remove_columns = dataset.column_names
        return dataset.map(
            tokenize_and_pack,
            batched=True,
            batch_size=1_000,
            remove_columns=remove_columns,
        )

    def get_train_eval_datasets(
        self, *, quick_run: bool = False, quick_examples: int = 4_000, eval_examples: int = 20_000
    ) -> tuple[datasets.Dataset | datasets.IterableDataset, datasets.Dataset | datasets.IterableDataset]:
        train, evaluation = self.get_train_eval_text_datasets(
            quick_run=quick_run,
            quick_examples=quick_examples,
            eval_examples=eval_examples,
        )
        return self._tokenize_and_pack(train), self._tokenize_and_pack(evaluation)

    @staticmethod
    def _preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    @staticmethod
    def _compute_metrics(prediction) -> dict[str, float]:
        predicted_ids = prediction.predictions
        if isinstance(predicted_ids, tuple):
            predicted_ids = predicted_ids[0]
        labels = prediction.label_ids
        active = labels != -100
        if not np.any(active):
            return {"masked_accuracy": 0.0, "masked_tokens": 0.0}
        accuracy = (predicted_ids[active] == labels[active]).mean()
        return {"masked_accuracy": float(accuracy), "masked_tokens": float(active.sum())}

    def get_training_args(
        self,
        *,
        max_steps: int,
        evaluation_fraction: float = 0.1,
        push_to_hub: bool = True,
    ) -> TrainingArguments:
        """Make percentage-based logging/evaluation/checkpoint intervals explicit."""

        if max_steps <= 0:
            raise ValueError("max_steps must be positive for the streaming full corpus")

        if not 0 < evaluation_fraction < 1:
            raise ValueError("evaluation_fraction must be a fraction in the range (0, 1)")

        hardware = self.hardware
        return TrainingArguments(
            output_dir=str(self.output_dir),
            max_steps=max_steps,
            learning_rate=5e-4,
            lr_scheduler_type="cosine",
            # Transformers interprets floats in [0, 1) as fractions of the total
            # number of steps. This is the non-deprecated warmup_ratio equivalent.
            warmup_steps=0.02,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-6,
            optim="adamw_torch_fused" if hardware.device == "cuda" else "adamw_torch",
            per_device_train_batch_size=hardware.per_device_train_batch_size,
            per_device_eval_batch_size=hardware.per_device_eval_batch_size,
            gradient_accumulation_steps=hardware.gradient_accumulation_steps,
            gradient_checkpointing=hardware.gradient_checkpointing,
            bf16=hardware.use_bf16,
            fp16=hardware.use_fp16,
            # Trainer's tf32 flag still writes PyTorch's legacy allow_tf32
            # setting. Inductor reads the newer precision API and rejects that
            # mixed state. BF16 training does not need TF32, so leave this unset
            # for compiled runs and retain TF32 for eager FP32-capable runs.
            tf32=None if self.compile_model else hardware.use_tf32,
            auto_find_batch_size=hardware.device == "cuda",
            dataloader_num_workers=hardware.dataloader_num_workers,
            dataloader_pin_memory=hardware.device == "cuda",
            dataloader_persistent_workers=hardware.dataloader_num_workers > 0,
            dataloader_prefetch_factor=4 if hardware.dataloader_num_workers > 0 else None,
            # torch.compile wraps ``forward`` as (*args, **kwargs), preventing
            # Trainer from discovering the normal input_ids/attention_mask
            # signature. The packed dataset contains only model inputs.
            remove_unused_columns=False,
            # The same generic compiled signature also hides the MLM label
            # field. Without this Trainer runs evaluation but cannot collect
            # eval_loss, accuracy, or perplexity.
            label_names=["labels"],
            logging_strategy="steps",
            logging_steps=evaluation_fraction,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=evaluation_fraction,
            save_strategy="steps",
            save_steps=evaluation_fraction,
            save_total_limit=3,
            prediction_loss_only=False,
            eval_accumulation_steps=8,
            report_to="none",
            run_name=self.hub_model_id.replace("/", "-"),
            seed=self.seed,
            data_seed=self.seed,
            push_to_hub=push_to_hub,
            hub_model_id=self.hub_model_id if push_to_hub else None,
            hub_strategy="every_save" if push_to_hub else "end",
            hub_always_push=push_to_hub,
        )

    def get_trainer(
        self,
        train_dataset: datasets.Dataset | datasets.IterableDataset,
        eval_dataset: datasets.Dataset | datasets.IterableDataset,
        training_args: TrainingArguments,
    ) -> Trainer:
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15,
            pad_to_multiple_of=8 if self.hardware.device == "cuda" else None,
        )
        return MLMTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=collator,
            compute_metrics=self._compute_metrics,
            preprocess_logits_for_metrics=self._preprocess_logits_for_metrics,
        )

    def train(
        self,
        *,
        max_steps: int = 250_000,
        evaluation_fraction: float = 0.1,
        quick_run: bool = False,
        quick_examples: int = 4_000,
        eval_examples: int = 20_000,
        push_to_hub: bool = True,
        resume_from_checkpoint: str | None = None,
    ) -> Trainer:
        """Train SANOLM and upload each periodic checkpoint when configured.

        Trainer logs ``loss`` at every interval (training loss), then ``eval_loss``,
        masked-token accuracy, and perplexity on the held-out test stream at the
        same 10%/20% cadence.  With hub uploads enabled every saved checkpoint is
        pushed to ``jangedoo/sanolm-v1-base``.
        """

        set_seed(self.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_dataset, eval_dataset = self.get_train_eval_datasets(
            quick_run=quick_run,
            quick_examples=quick_examples,
            eval_examples=eval_examples,
        )
        args = self.get_training_args(
            max_steps=max_steps,
            evaluation_fraction=evaluation_fraction,
            push_to_hub=push_to_hub,
        )
        trainer = self.get_trainer(train_dataset, eval_dataset, args)
        print(f"SANOLM hardware settings: {asdict(self.hardware)}")
        trainer.log({"model_parameters": self.parameter_count})
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model()
        if push_to_hub:
            trainer.push_to_hub()
        return trainer

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the SANOLM ModernBERT encoder")
    parser.add_argument("--output-dir", default="trainer_output/sanolm-v1-base")
    parser.add_argument("--max-steps", type=int, default=250_000)
    parser.add_argument("--eval-fraction", type=float, choices=(0.1, 0.2), default=0.1)
    parser.add_argument("--quick-run", action="store_true")
    parser.add_argument("--quick-examples", type=int, default=4_000)
    parser.add_argument("--eval-examples", type=int, default=20_000)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--no-push-to-hub", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    recipe = SANOLMBase(output_dir=args.output_dir, compile_model=not args.no_compile)
    print(f"SANOLM parameters: {recipe.parameter_count:,}")
    print(f"Hardware settings: {asdict(recipe.hardware)}")
    recipe.train(
        max_steps=args.max_steps,
        evaluation_fraction=args.eval_fraction,
        quick_run=args.quick_run,
        quick_examples=args.quick_examples,
        eval_examples=args.eval_examples,
        push_to_hub=not args.no_push_to_hub,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
