# src/model.py
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data import BoltHoleDataset
from config import config
from src.data import load_data
from src.utils import train_one_epoch, validate

class BoltHoleModel(nn.Module):
    def __init__(self,num_classes = 3):
        super(BoltHoleModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base_model.last_channel, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def train_model(data_dir,model_path,config):
    # 加载数据
    transform = transforms.Compose([
        transforms.resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = BoltHoleDataset(data_dir ,transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    
    model = BoltHoleModel(num_classes=config['num_classes']).to(config['device'])
    critirion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    for epoch in range(config['num_epochs']):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = critirion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() 
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")
        
        if (epoch+1) % config['save_freq'] == 0:
            torch.save(model.state_dict(), f"{model_path}/checkpoints/model_{epoch+1}.pth")
        torch.save(model.state_dict(), f"{model_path}/model_final.pth")
        print(f"Model saved to {model_path}/model_final.pth")
        
    

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    train_model('data/processed', 'models/final_model.pt', config)
    

# with LIDAR huh?