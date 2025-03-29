import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-3B-Instruct-Braille")

def read_text_file(file_path):
    """Reads the entire contents of a UTF-8 text file and returns it as a string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
import re

def numeric_key(prefix):
    """
    Convert '535_1' into a tuple like (535, 1),
    so Python sorts based on numeric value rather than string value.
    """
    parts = prefix.split('_')
    numeric_parts = []
    for part in parts:
        # If it's purely digits, convert to int
        if part.isdigit():
            numeric_parts.append(int(part))
        else:
            # Otherwise, leave as string (in case of special cases)
            numeric_parts.append(part)
    return tuple(numeric_parts)


def main():
    # initialize an empty list to store the JSON records
    parser = argparse.ArgumentParser(description="Converts a directory of .txt files into a JSON dataset.")
    parser.add_argument("--base_dir", type=str, default="./data/collected-datasets", help="The directory containing your .txt files")
    parser.add_argument("--output_dir", type=str, default="./data", help="The directory to write the output JSON file")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="The ratio of training data to test data")
    parser.add_argument("--curoflen", type=int, default=4096, help="The current offset length")
    args = parser.parse_args()
    records = []

    # The directory containing your .txt files

    # A map from prefix -> {"braille": "...", "chn": "..."}
    #   e.g. prefix "535_1" => {
    #       "braille": "./data/.../535_1_braille.txt",
    #       "chn": "./data/.../535_1_chn.txt"
    #   }

    i = 0

    for book_dir in tqdm(os.listdir(args.base_dir),desc="Processing books"):
        book_path = os.path.join(args.base_dir, book_dir)   

        file_map = {}
        # 1) Collect braille and chn files in the directory
        for fname in os.listdir(book_path):
            fpath = os.path.join(book_path, fname)

            # We only care about actual files (not directories)
            if os.path.isfile(fpath):
                if fname.endswith("_braille.txt"):
                    prefix = fname.replace("_braille.txt", "")
                    file_map.setdefault(prefix, {})["braille"] = fpath
                elif fname.endswith("_chn.txt"):
                    prefix = fname.replace("_chn.txt", "")
                    file_map.setdefault(prefix, {})["chn"] = fpath

        # 2) Build a list of valid pairs (prefixes that have both braille & chn)
        file_pairs = []
        for prefix in sorted(file_map.keys(), key=numeric_key):
            paths = file_map[prefix]
            if "braille" in paths and "chn" in paths:
                file_pairs.append((paths["braille"], paths["chn"]))

        # 3) For each braille/chn pair, read contents and build JSON structure
        for braille_path, chn_path in file_pairs:
            braille_content = read_text_file(braille_path)
            chn_content = read_text_file(chn_path)

            # print(args.curoflen)
            if len(tokenizer.tokenize(braille_content))>=args.curoflen or len(tokenizer.tokenize(chn_content))>=args.curoflen:
                continue

            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个中国盲文翻译助手，请把通用盲文转换成为汉字"
                    },
                    {
                        "role": "user",
                        "content": (
                            # f"Translate the following Chinese braille into Chinese. {braille_path}\n\n"
                            f"请把一下通用盲文转换成为汉字.\n"
                            f"Braille content:\n{braille_content}"
                        )
                    },
                    {
                        "role": "assistant",
                        "content": (
                            # f"The corresponding Chinese data. {chn_path}\n\n"
                            # f"The corresponding Chinese data.\n"
                            f"对应的中文内容是:\n{chn_content}"
                        )
                    }
                ]
            }

            records.append(record)


    # 3.5) Split into train and test sets
    #   - 80% train, 20% test
    train_data = records[:int(len(records) * args.train_ratio)]
    test_data = records[int(len(records) * args.train_ratio):]

    # Remove any message with role "assistant" from each record in test_data
    # for record in test_data:
    #     record["messages"] = [msg for msg in record["messages"] if msg.get("role") != "assistant"]

    # 4) Convert to JSON and print/write to file
    # Print to console:
    output_json_train = json.dumps(train_data, indent=2, ensure_ascii=False)
    output_json_test = json.dumps(test_data, indent=2, ensure_ascii=False)
    # print(output_json_train)
    # print(output_json_test)
    
    # Or, write to file:
    with open(args.output_dir+"/parallel_data_train.json", "w", encoding="utf-8") as out_f:
        out_f.write(output_json_train)
    print("Wrote to parallel_data_train.json")
    
    with open(args.output_dir+"/parallel_data_test.json", "w", encoding="utf-8") as out_f:
        out_f.write(output_json_test)
    print("Wrote to parallel_data_test.json")

if __name__ == "__main__":
    main()