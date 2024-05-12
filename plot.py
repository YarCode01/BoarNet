import matplotlib.pyplot as plt
import json
EXPERIMENT_TIME = "10-05-2024 15-34-56"
with open(f"results/losses/{EXPERIMENT_TIME}/train_losses.json", 'r') as file:
    train_losses = json.load(file)

with open(F"results/losses/{EXPERIMENT_TIME}/test_losses.json", 'r') as file:
    test_losses = json.load(file)

plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Losses')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()