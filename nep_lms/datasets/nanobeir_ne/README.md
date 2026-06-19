Vibe Coded script to translate NanoBEIR dataset to Nepali. Keeping it here for reference.

Dataset available here: https://huggingface.co/datasets/jangedoo/NanoBEIR-ne


```
poetry run python nep_lms/datasets/nanobeir_ne/translate_nanobeir.py --env-file .env --output-dir nep_lms/datasets/nanobeir_ne/output --skip-existing --show-progress --parallel-requests 16 --log-llm-requests --repo-id "jangedoo/NanoBEIR-ne" --push-to-hub
```