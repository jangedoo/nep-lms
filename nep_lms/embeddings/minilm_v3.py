from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers import losses, training_args, util

nep_mini_lm_v3 = SentenceTransformer("jangedoo/all-MiniLM-L6-v2-nepali")
args = SentenceTransformerTrainingArguments(
    output_dir=None,
    num_train_epochs=3,
    max_steps=300,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=util.get_device_name() != "cpu",
    save_strategy="epoch",
    logging_steps=50,
    batch_sampler=training_args.BatchSamplers.NO_DUPLICATES,  # avoids identical positives in one batch
)
trainer = SentenceTransformerTrainer(
    model=nep_mini_lm_v3,
    args=args,
    train_dataset=exp.nepali_news_ds["train"].select_columns(["title", "excerpt"]),
    eval_dataset=exp.nepali_news_ds["test"].select_columns(["title", "excerpt"]),
    loss=losses.MultipleNegativesRankingLoss(nep_mini_lm_v3),
)
# trainer.train()
