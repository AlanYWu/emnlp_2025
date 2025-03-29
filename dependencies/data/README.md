# Braille-Chinese Parallel Dataset

This folder contains the training and evaluation data used for fine-tuning the Qwen2.5-3B-Instruct model on Braille-to-Chinese translation tasks.

## Contents

- `parallel_data_train.json`  
  Contains aligned Braille-Chinese training pairs for supervised fine-tuning.

- `parallel_data_test.json`  
  Contains held-out validation/test pairs for model evaluation.

- `collected_dataset/`  
  A collection of raw Braille-Chinese sentence pairs gathered from various online resources. This folder includes unfiltered or less-processed data and serves as the source pool from which `parallel_data_train.json` and `parallel_data_test.json` were curated.

## Format

Each `.json` file is a list of objects. Each object typically contains:
```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "你是一个中国盲文翻译助手，请把通用盲文转换成为汉字"
      },
      {
        "role": "user",
        "content": "请把一下通用盲文转换成为汉字.\nBraille content:\n"
      },
      {
	"role": "assistant",
	"content": "对应的中文内容是:\n"
      }
    ]
  }
]
