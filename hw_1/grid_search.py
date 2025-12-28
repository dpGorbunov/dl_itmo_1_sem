import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

df = pd.read_csv('cybersecurity.csv')
X = df.drop('Class', axis=1).values
y = df['Class'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

class MalwareDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = MalwareDataset(X_train_scaled, y_train)
val_dataset = MalwareDataset(X_val_scaled, y_val)
test_dataset = MalwareDataset(X_test_scaled, y_test)

class BaselineMLP(nn.Module):
    """Базовая MLP: 15 -> 128 -> 64 -> 1"""
    def __init__(self, input_size=15, hidden1=128, hidden2=64):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class GridSearchMLP(nn.Module):
    def __init__(self, input_size=15, hidden1=128, hidden2=64, use_bn=False, dropout_p=0.0):
        super(GridSearchMLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1) if use_bn else nn.Identity()
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2) if use_bn else nn.Identity()
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        self.fc3 = nn.Linear(hidden2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        return self.sigmoid(self.fc3(x))

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
    return running_loss / len(dataloader.dataset)

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * features.size(0)
    return running_loss / len(dataloader.dataset)

# Создаю baseline DataLoader (bs=32) до цикла
BATCH_SIZE = 32
baseline_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
baseline_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
baseline_test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Проверка размерности батча
_ = next(iter(baseline_train_loader))

configs = []

# Baseline
configs.append({
    'name': 'Baseline, bs=32',
    'use_bn': False,
    'dropout_p': 0.0,
    'lr': 0.001,
    'batch_size': 32
})

# Dropout с разными p
for p in [0.03, 0.05, 0.055, 0.07, 0.1]:
    configs.append({
        'name': f'Drop(p={p}), bs=32',
        'use_bn': False,
        'dropout_p': p,
        'lr': 0.001,
        'batch_size': 32
    })

# Разные batch_size
for bs in [64, 96]:
    configs.append({
        'name': f'Drop(p=0.05), bs={bs}',
        'use_bn': False,
        'dropout_p': 0.05,
        'lr': 0.001,
        'batch_size': bs
    })

# BatchNorm
configs.append({
    'name': 'BatchNorm, bs=32',
    'use_bn': True,
    'dropout_p': 0.0,
    'lr': 0.001,
    'batch_size': 32
})

# BatchNorm + Dropout
for p in [0.03, 0.05]:
    configs.append({
        'name': f'BN+Drop(p={p}), bs=32',
        'use_bn': True,
        'dropout_p': p,
        'lr': 0.001,
        'batch_size': 32
    })

results = []
NUM_EPOCHS = 150

print(f"\nВсего конфигураций: {len(configs)}\n")

for i, config in enumerate(configs, 1):
    print(f"[{i}/{len(configs)}] {config['name']}")

    # Использую baseline DataLoader для bs=32, создаю новые для других размеров
    if config['batch_size'] == 32:
        train_loader = baseline_train_loader
        val_loader = baseline_val_loader
        test_loader = baseline_test_loader
    else:
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    # Использую BaselineMLP для baseline, GridSearchMLP для остальных
    if 'Baseline' in config['name']:
        model = BaselineMLP().to(device)
    else:
        model = GridSearchMLP(use_bn=config['use_bn'], dropout_p=config['dropout_p']).to(device)

    # Проверка forward pass
    if config['batch_size'] == 32:
        _ = next(iter(train_loader))

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    start_time = time.time()

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features.to(device))
            y_pred.extend((outputs.cpu().numpy() > 0.5).astype(int))
            y_true.extend(labels.numpy().astype(int))

    test_acc = accuracy_score(np.array(y_true).flatten(), np.array(y_pred).flatten())
    elapsed = time.time() - start_time

    min_val_loss_epoch = np.argmin(val_losses)

    results.append({
        'config': config['name'],
        'use_bn': config['use_bn'],
        'dropout_p': config['dropout_p'],
        'lr': config['lr'],
        'batch_size': config['batch_size'],
        'epochs': NUM_EPOCHS,
        'test_acc': float(test_acc),
        'train_loss': float(train_losses[-1]),
        'val_loss': float(val_losses[-1]),
        'min_val_loss': float(min(val_losses)),
        'min_val_loss_epoch': int(min_val_loss_epoch),
        'time_sec': float(elapsed)
    })

    print(f"  Acc: {test_acc:.4f}, Val Loss: {val_losses[-1]:.4f} (min: {min(val_losses):.4f} @ ep {min_val_loss_epoch}), Time: {elapsed:.1f}s\n")

# Сохранение в CSV
df_results = pd.DataFrame(results)
df_results.to_csv('grid_search_results.csv', index=False)
print(f"\nРезультаты сохранены в grid_search_results.csv")

# Вывод результатов
print("="*80)
print("РЕЗУЛЬТАТЫ GRID SEARCH")
print("="*80)

results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)

print(f"\n{'Конфигурация':<30} {'Test Acc':<12} {'Val Loss':<12} {'Time (s)'}")
print("-" * 70)
for r in results_sorted:
    print(f"{r['config']:<30} {r['test_acc']:<12.4f} {r['val_loss']:<12.4f} {r['time_sec']:<12.1f}")

print(f"\n{'='*80}")
print(f"ЛУЧШАЯ КОНФИГУРАЦИЯ: {results_sorted[0]['config']}")
print(f"Test Accuracy: {results_sorted[0]['test_acc']:.4f}")
print(f"Val Loss: {results_sorted[0]['val_loss']:.4f}")
print(f"Min Val Loss: {results_sorted[0]['min_val_loss']:.4f} (epoch {results_sorted[0]['min_val_loss_epoch']})")
print(f"use_bn: {results_sorted[0]['use_bn']}, dropout_p: {results_sorted[0]['dropout_p']}, lr: {results_sorted[0]['lr']}, batch_size: {results_sorted[0]['batch_size']}")
print(f"{'='*80}")
