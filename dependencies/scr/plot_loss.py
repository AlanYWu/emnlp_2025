import json
import matplotlib.pyplot as plt

# Initialize lists
steps = []
train_losses = []
eval_steps = []
eval_losses = []

# Read the JSONL log file
with open('saves/Qwen2.5-3B-Instruct-Braille/train_braille/trainer_log.jsonl', 'r') as f:
    for line in f:
        log = json.loads(line)
        # Collect training loss if present
        if 'loss' in log:
            steps.append(log['current_steps'])
            train_losses.append(log['loss'])
        # Collect evaluation loss if present
        if 'eval_loss' in log:
            eval_steps.append(log['current_steps'])
            eval_losses.append(log['eval_loss'])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(steps, train_losses, label='Training Loss', marker='o')
plt.plot(eval_steps, eval_losses, label='Evaluation Loss', marker='x', color='red')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.grid(True)
plt.show()


plt.savefig('loss_plot.png', dpi=300)