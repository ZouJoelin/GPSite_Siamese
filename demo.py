import os
from tqdm import tqdm

import torch
from torch_geometric.loader import DataLoader

from data import *
from model import *


info = torch.load("./data/dataset.pt")
train_dataset = SiameseProteinGraphDataset(info, feature_path="./data/", radius=15)

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

print(f"train_dataset: {train_dataset.__len__()}")  # total_data * (fold-1)/fold
print(f"train_dataloader: {train_dataloader.__len__()}")  # num_samples / batch_size


model = SiameseGPSite()

learning_rate = 1e-3
beta12 = (0.9, 0.99)
num_epochs = 25
epochs = 25

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=beta12, weight_decay=1e-5, eps=1e-5)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dataloader), epochs=epochs)  # ???

# test
train_data = next(iter(train_dataloader))
model.eval()
with torch.no_grad():
    # train_data = train_data.to(device)
    wt_graph, mut_graph, y = train_data
    print(wt_graph)
    print(mut_graph)

    pred = model(wt_graph, mut_graph)

    print(f"Predicted: {pred}\nActual: {y}")
    # print((pred == y))
    # print(((pred == y)).sum())
    









