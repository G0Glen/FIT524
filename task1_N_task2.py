import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from a3_mnist import Lenet  # The pre-trained target model (LeNet) provided by instructor

# === 0. Defense Configuration ===
defense_on = True                   # Whether to enable defense mechanisms 
noise_std = 0.3                    # Standard deviation of Gaussian noise added to softmax probabilities

# === 1. Define the Attacker’s Stolen Model ===
class StealModel(nn.Module):
    """
    A convolutional neural network (CNN) designed to mimic the behavior of a target model.
    
    Purpose:
        Used by the attacker to train a substitute model by querying the black-box target model.
    
    Architecture:
        - 2 Convolutional Layers
        - Max Pooling
        - Dropout to avoid overfitting
        - 2 Fully Connected Layers
        - Log-Softmax for classification
    """
    def __init__(self):
        super(StealModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)     # First conv layer: input 1 channel (grayscale), 32 filters
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)    # Second conv layer: 32 → 64 channels
        self.pool = nn.MaxPool2d(2, 2)                  # 2x2 max pooling
        self.dropout = nn.Dropout(0.2)                  # Dropout layer with 20% dropout
        self.fc1 = nn.Linear(64 * 7 * 7, 128)           # First FC layer: 3136 → 128
        self.fc2 = nn.Linear(128, 10)                   # Output layer: 128 → 10 classes (digits 0–9)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (Tensor): Input batch of images with shape (batch_size, 1, 28, 28)
        
        Returns:
            Tensor: Log-Softmax outputs (log probabilities for each class)
        """
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 → ReLU → MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 → ReLU → MaxPool
        x = self.dropout(x)                   # Dropout applied to reduce overfitting
        x = x.view(-1, 64 * 7 * 7)            # Flatten for FC layer
        x = F.relu(self.fc1(x))               # FC1 with ReLU activation
        x = self.fc2(x)                       # FC2 outputs raw logits
        return F.log_softmax(x, dim=1)        # Convert logits to log-probabilities

# === 2. Simulate Target Model Query defense ===
def query_target_model(images, defense=defense_on, noise_std=noise_std):
    """
    Simulates black-box querying of the target model.
    
    If defense is enabled:
        - Adds Gaussian noise to the output probabilities
        - Normalizes the result to maintain a valid distribution

    Args:
        images (Tensor): A batch of MNIST images
        defense (bool): Whether to apply the defense strategy
        noise_std (float): Standard deviation of added noise
    
    Returns:
        Tensor: The softmax (or noised) probabilities from the target model
    """
    if not hasattr(query_target_model, "model"):
        model = Lenet()
        model.load_state_dict(torch.load('target_model.pth'))
        model.eval()
        print("Target model loaded.")
        query_target_model.model = model

    with torch.no_grad():
        logits = query_target_model.model(images)
        probs = F.softmax(logits, dim=1)

        if defense:
            noise = torch.randn_like(probs) * noise_std
            noisy_probs = probs + noise
            noisy_probs = torch.clamp(noisy_probs, min=1e-6, max=1.0)
            noisy_probs = noisy_probs / noisy_probs.sum(dim=1, keepdim=True)
            return noisy_probs
        else:
            return probs

# === 3. Pseudo Label Generation with Optional Label Flipping ===
def get_pseudo_labels(probs):
    """
    Generate flipped labels using the 'least confidence' strategy.
    
    This function returns, for each sample in the batch, the class index 
    with the *lowest* predicted probability — intentionally assigning the 
    least confident (most likely wrong) label.

    Args:
        probs (Tensor): A tensor of shape (batch_size, num_classes), 
                        representing the softmax probabilities output by the target model.

    Returns:
        Tensor: A tensor of shape (batch_size,) containing the flipped labels.
    """
    return torch.argmin(probs, dim=1)

# === 4. Load MNIST Data ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std normalization for MNIST
])
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# === 5. Initialize Stolen Model, Optimizer, and Loss ===
steal_model = StealModel()
criterion = nn.NLLLoss()  # Negative log likelihood for log-softmax output
optimizer = optim.Adam(steal_model.parameters(), lr=0.001)

# === 6. Train the Stolen Model Using Pseudo-Labels from Target ===
print(f"\n--- Training stolen model ---")
print(f"Defense Enabled: {defense_on}, Noise STD: {noise_std}\n")

for epoch in range(1, 6):  # Train for 5 epochs
    steal_model.train()
    running_loss = 0.0
    for batch_idx, (images, _) in enumerate(train_loader):
        outputs = query_target_model(images)               # Step 1: Get target model's output
        pseudo_labels = get_pseudo_labels(outputs)         # Step 2: Generate pseudo labels (possibly flipped)
        preds = steal_model(images)                        # Step 3: Forward pass of stolen model
        loss = criterion(preds, pseudo_labels)             # Step 4: Loss calculation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Print progress every 640 batches (~every epoch)
        if batch_idx % 640 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
    print(f"End of Epoch {epoch} - Avg Loss: {running_loss / len(train_loader):.4f}")

# === 7. Evaluate Stolen Model on True Labels ===
steal_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = steal_model(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"\nSteal Model Accuracy on Test Set: {100 * correct / total:.2f}%")

# === 8. Measure Agreement Between Stolen and Target Models ===
agreement = 0
total = 0
with torch.no_grad():
    for images, _ in DataLoader(train_data, batch_size=64):
        target_preds = torch.argmax(query_target_model(images), dim=1)
        steal_preds = torch.argmax(steal_model(images), dim=1)
        agreement += (target_preds == steal_preds).sum().item()
        total += target_preds.size(0)
print(f"Prediction Agreement with Target Model: {100 * agreement / total:.2f}%")
