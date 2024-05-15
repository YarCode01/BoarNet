import matplotlib.pyplot as plt
import json
EXPERIMENT_TIME = "15-05-2024 16-55-34"
with open(f"results/losses/{EXPERIMENT_TIME}/train_losses.json", 'r') as file:
    train_losses = json.load(file)

with open(F"results/losses/{EXPERIMENT_TIME}/test_losses.json", 'r') as file:
    test_losses = json.load(file)

plt.plot(range(1, 28), train_losses[:27], label='Train Loss')
plt.plot(range(1, 28), test_losses[:27], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Losses')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()