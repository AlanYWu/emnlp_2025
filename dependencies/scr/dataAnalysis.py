import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-3B-Instruct-Braille")

# Load the JSON file
with open("./data/parallel_data_train.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract user message lengths
user_message_lengths = [
    len(tokenizer.tokenize(msg["content"])) for conversation in data for msg in conversation["messages"] if msg["role"] == "user"
]

# Extract assistant message lengths
assistant_message_lengths = [
    len(tokenizer.tokenize(msg["content"])) for conversation in data for msg in conversation["messages"] if msg["role"] == "assistant"
]

# Plot histograms
plt.figure(figsize=(10, 5))
plt.hist(user_message_lengths, bins=30, alpha=0.5, label="User Messages", edgecolor="black")
plt.hist(assistant_message_lengths, bins=30, alpha=0.5, label="Assistant Messages", edgecolor="black")
plt.xlabel("Message Length (characters)")
plt.ylabel("Frequency")
plt.title("Histogram of Message Lengths")
plt.legend()

# Save the histogram
plt.savefig("./src/message_length_histogram_after_processing.png", dpi=300)


print(f"Number of user messages: {len(user_message_lengths)}")
print(f"Number of assistant messages: {len(assistant_message_lengths)}")
print(f"Total length of user messages: {sum(user_message_lengths)} characters")
print(f"Total length of assistant messages: {sum(assistant_message_lengths)} characters")

# Compute average lengths
average_user_length = sum(user_message_lengths) / len(user_message_lengths) if user_message_lengths else 0
average_assistant_length = sum(assistant_message_lengths) / len(assistant_message_lengths) if assistant_message_lengths else 0

# Print results
print(f"Average input length of user messages: {average_user_length:.2f} characters")
print(f"Average input length of assistant messages: {average_assistant_length:.2f} characters")

# Define percentile values
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Compute percentiles for both user and assistant messages
user_percentile_values = np.percentile(user_message_lengths, percentiles)
assistant_percentile_values = np.percentile(assistant_message_lengths, percentiles)

# Print percentiles
print("User Message Length Percentiles:")
for p, value in zip(percentiles, user_percentile_values):
    print(f"{p}%: {value:.2f} characters")

print("\nAssistant Message Length Percentiles:")
for p, value in zip(percentiles, assistant_percentile_values):
    print(f"{p}%: {value:.2f} characters")