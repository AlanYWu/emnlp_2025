# Model Artifacts

This folder contains all model-related assets used for training, evaluation, and deployment of the Braille-to-Chinese fine-tuned models.

---

## Directory Structure

### `Qwen2.5-3B-Instruct/`
- **Description**: Original pretrained instruction-tuned model downloaded from [Hugging Face](https://huggingface.co).
- **Use**: Serves as the base model for fine-tuning.

### `Qwen2.5-3B-Instruct-Braille/`
- **Description**: The fine-tuned model on Braille-to-Chinese parallel data.
- **How it was generated**: This model was created using the script `addSpecialTokens.py` to augment the tokenizer and training setup.

### `train_checkpoints/`
- **Description**: Stores training outputs, including:
  - Checkpoints (`checkpoint-*` directories)
  - `trainer_state.json`, `pytorch_model.bin`, etc.
  - Training logs
- **Note**: This directory is useful for resuming training or analyzing model progress.

### `test_results/`
- **Description**: Contains evaluation logs and outputs from model testing on held-out data.

---

## Notes

- Checkpoints are **not tracked by Git** to reduce repo size. If needed, upload to Hugging Face or external storage and link in your documentation.
- The fine-tuning configuration files are located under `dependencies/config/`.

---

## Reproducibility

To reproduce the fine-tuned model:
1. Download the base model (`Qwen2.5-3B-Instruct`) from Hugging Face.
2. Apply special token modifications using `addSpecialTokens.py`.
3. Fine-tune using the training scripts in `dependencies/scripts/`.
