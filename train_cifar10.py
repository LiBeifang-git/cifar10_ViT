import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import subprocess
from torch.utils.tensorboard import SummaryWriter  
from vit import vit_base_patch16_224

#def keep_mac_awake():
#    pid = os.getpid() 
#    subprocess.Popen(['caffeinate', '-i', '-w', str(pid)])
#keep_mac_awake()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--model", type=str, default="vit")
args = parser.parse_args()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

if args.model.lower().startswith("vit"):
    model = vit_base_patch16_224(num_class=10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
steps_per_epoch = len(train_loader)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,                
    epochs=args.epochs,             
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,                  
    anneal_strategy='cos'         
)

def train_and_eval():
    best_acc = 0.0
    os.makedirs("./ckpt", exist_ok=True)

    log_dir = f"./runs/{args.model}_bs{args.batch_size}_lr{args.lr}"
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Train")
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
     
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            global_step += 1
            
        train_loss = running_loss / len(trainset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        print(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

        writer.add_scalar('Loss/Train_Epoch', train_loss, epoch + 1)
        writer.add_scalar('Accuracy/Test_Epoch', test_acc, epoch + 1)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"./ckpt/{args.model}_best.pth")
            
    writer.close()

if __name__ == '__main__':
    print("使用模型:", args.model)
    try:
        train_and_eval()
    except KeyboardInterrupt:
        os.makedirs("./ckpt", exist_ok=True)
        torch.save(model.state_dict(), "./ckpt/interrupted_emergency.pth")