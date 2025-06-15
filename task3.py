import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# === Step 1: Define Target & Shadow Model (LeNet) ===
class ShadowNet(nn.Module):
    def __init__(self):
        super(ShadowNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# === Step 2: Load and Split MNIST for Shadow Training ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
member_data, nonmember_data = random_split(dataset, [30000, 30000])
member_loader = DataLoader(member_data, batch_size=128, shuffle=True)
nonmember_loader = DataLoader(nonmember_data, batch_size=128, shuffle=True)

# === Step 3: Train Shadow Model ===
model = ShadowNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("🔧 Training Shadow Model...")
model.train()
for epoch in range(5):
    total_loss = 0
    for x, y in member_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(member_loader):.4f}")

# === Step 4: Extract Softmax Confidence and Additional Features ===
def extract_features(model, dataloader, label):
    model.eval()
    X, Y = [], []
    with torch.no_grad():
        for images, _ in dataloader:
            probs = model(images)
            confidence, predicted = probs.max(dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
            loss = criterion(probs.log(), predicted)
            for c, e, l in zip(confidence, entropy, loss.repeat(len(images))):
                X.append([c.item(), e.item(), l.item()])
                Y.append(label)
    return X, Y

member_X, member_Y = extract_features(model, member_loader, 1)
nonmember_X, nonmember_Y = extract_features(model, nonmember_loader, 0)

X_total = member_X + nonmember_X
Y_total = member_Y + nonmember_Y

print("✅ Features (confidence + entropy + loss) extracted for attack model.")

# === Step 5: Train Attack Model ===
attack_model = LogisticRegression(max_iter=1000)
attack_model.fit(X_total, Y_total)

# === Step 6: Evaluation ===
preds = attack_model.predict(X_total)
acc = accuracy_score(Y_total, preds)
prec = precision_score(Y_total, preds)
rec = recall_score(Y_total, preds)
f1 = f1_score(Y_total, preds)

print("\n📊 Attack Model Evaluation:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")