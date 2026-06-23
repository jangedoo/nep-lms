import json
import re

import datasets
import tokenizers
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

SEED = 100

en = (
    datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name='sample-10BT',
        split="train",
        streaming=True,
    )
    .select_columns(["text"])
).shuffle(seed=SEED, buffer_size=10000)



ne = datasets.load_dataset("jangedoo/nepali-corpus", split='train', streaming=True).shuffle(seed=SEED, buffer_size=10000).select_columns(['text'])
ne_mostly_romanized = datasets.load_dataset("jangedoo/nepali-reddit", name="comments", split='nepalsocial', streaming=True).shuffle(seed=SEED, buffer_size=10000).rename_column('body', 'text').select_columns(['text'])

mixed = datasets.interleave_datasets(
    [ne, ne_mostly_romanized, en],
    probabilities=[0.60, 0.20, 0.20],
    seed=SEED,
    stopping_strategy="all_exhausted",
)

def batch_iterator(batch_size=1000, max_chars=500_000_000):
    tok_dataset = mixed.select_columns("text")
    seen_chars = 0
    for batch in tok_dataset.iter(batch_size):
        for text in batch['text']:
            seen_chars += len(text.strip())

        yield batch["text"]

        if seen_chars >= max_chars:
            break


# https://tokka-bench.streamlit.app/
# according to this gemma3 has best efficiency for Nepali
# so we try to replicate its tokenizer

tokenizer = Tokenizer(
    models.BPE(
        unk_token="<unk>",
        byte_fallback=True,
        fuse_unk=False,
    )
)
tokenizer.normalizer = normalizers.Replace(" ", "▁")

tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
      pre_tokenizers.Split(
          tokenizers.Regex(r"\r\n|\r|\n"),
          behavior="isolated",
      ),
      pre_tokenizers.Split(
          pattern="▁",
          behavior="merged_with_next",
          invert=False,
      ),
  ])

tokenizer.decoder = decoders.Sequence(
    [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
)

tokenizer.post_processor = processors.TemplateProcessing(
    single="<bos> $A",
    pair="<bos> $A <eos> $B:1",
    special_tokens=[
        ("<bos>", 1),
        ("<eos>", 2),
    ],
)

special_tokens = [
    "<unk>",
    "<bos>",
    "<eos>",
    "<pad>",
    "<mask>",
]
byte_fallback_tokens = [f"<0x{i:02X}>" for i in range(256)]
all_special_tokens = special_tokens + byte_fallback_tokens
trainer = trainers.BpeTrainer(
    vocab_size=42000,
    min_frequency=2,
    special_tokens=all_special_tokens,
    show_progress=True,
)

_ = tokenizer.train_from_iterator(batch_iterator(5000, max_chars=5_000_000_000), trainer)

save_path = "./tokenizer.json"
tokenizer.save(save_path)

# according to chat gpt, we need to remove is_special marker for byte fallback tokens
# otherwise those tokens won't get decoded


with open(save_path, "r", encoding="utf-8") as f:
    data = json.load(f)

byte_token_re = re.compile(r"^<0x[0-9A-F]{2}>$")

for tok in data.get("added_tokens", []):
    if byte_token_re.match(tok["content"]):
        tok["special"] = False

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

tokenizer = Tokenizer.from_file(save_path)

texts = [
    "how are you",
    "this is nice",
    "mero nam sanjaya ho",
    "तपाई कस्तो हुनुहुन्छ",
    "the world has become quite advanced now."
    # small_texts[0][:20],
]
for text in texts:
    print(text)
    encoded = tokenizer.encode(text)
    print(encoded.ids, encoded.tokens)
    print(tokenizer.decode(encoded.ids))
    print()


# save
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<unk>",
    bos_token="<bos>",
    eos_token="<eos>",
    pad_token="<pad>",
    mask_token="<mask>",

    model_max_length=512,
)
hf_tokenizer.push_to_hub("jangedoo/sanolm-tokenizer")
print("Finished")