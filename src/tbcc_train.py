import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from models.tbcc.transformer import TransformerModel
from utils import read_pickle
from metrics import pearson_corr_v2 as pearson_corr


class TBCCDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_json(data_path, orient="records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "id": torch.tensor(self.data.iloc[idx]["id"]).long(),
            "x": torch.tensor(self.data.iloc[idx]["repr"]).long(),
            "y": torch.tensor([self.data.iloc[idx]["output"]]).float(),
        }


def collate_fn(batch, max_seq_length=None):
    id_list = [item['id'] for item in batch]
    x_list = [item['x'] for item in batch]
    y_list = [item['y'] for item in batch]

    # Truncate if necessary and pad each sequence to max_seq_length
    if max_seq_length is not None:
        x_padded = []
        for x in x_list:
            x = x[:max_seq_length]  # Truncate
            pad_size = max_seq_length - len(x)  # Calculate padding size
            if pad_size > 0:
                # Pad sequence to the desired length
                x = F.pad(x, (0, pad_size), 'constant', 0)
            x_padded.append(x)
        x_padded = torch.stack(x_padded)
    else:
        x_padded = pad_sequence(x_list, batch_first=True, padding_value=0)

    y_stacked = torch.stack(y_list)
    id_stacked = torch.stack(id_list)

    return x_padded, y_stacked, id_stacked



# Training loop
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch, id_batch in data_loader:
        X_batch, y_batch, id_batch = X_batch.to(device), y_batch.to(device), id_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)


# Evaluation function
def evaluate(model, data_loader, criterion, device):
    # P-CORR
    id_list = []
    y_pred_list = []
    y_true_list = []

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch, id_batch in data_loader:
            X_batch, y_batch, id_batch = X_batch.to(device), y_batch.to(device), id_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            id_list.extend(id_batch.detach().cpu().squeeze().tolist())
            y_pred_list.extend(outputs.detach().cpu().squeeze().tolist())
            y_true_list.extend(y_batch.detach().cpu().squeeze().tolist())

            total_loss += loss.item()
    return total_loss / len(data_loader), [id_list, y_pred_list, y_true_list]


def training(args):
    label_scaler = read_pickle(args.scaler_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = TBCCDataset(data_path=args.train_file)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, max_seq_length=args.max_seq_length))

    test_dataset = TBCCDataset(data_path=args.test_file)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, max_seq_length=args.max_seq_length))

    for batch in train_loader:
        print(f"[TRAINING] X: {batch[0].shape}, Y: {batch[1].shape}")
        break

    for batch in test_loader:
        print(f"[TESTING] X: {batch[0].shape}, Y: {batch[1].shape}")
        break

    net = TransformerModel(args.vocab_size, args.embed_dim, args.num_heads, args.ff_dim, args.num_transformer_block, args.max_seq_length, args.num_classes)
    net = net.to(device)

    # Cross Entropy loss
    criterion = nn.MSELoss()

    # Optimizer
    # optimizer = optim.SGD(net.parameters(), lr=LEARN_RATE)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    predictions = []
    metrics = {"train_loss": [], "test_loss": [], "test_p_corr": []}
    for epoch in range(args.epochs):
        train_loss = train_epoch(net, train_loader, criterion, optimizer, device)
        test_loss, [id_list, y_pred_list, y_true_list] = evaluate(net, test_loader, criterion, device)

        y_pred_list = label_scaler.inverse_transform(np.array(y_pred_list).reshape(1, -1))
        y_true_list = label_scaler.inverse_transform(np.array(y_true_list).reshape(1, -1))
        test_p_corr = pearson_corr(torch.tensor(y_pred_list).squeeze(), torch.tensor(y_true_list).squeeze())

        predictions = [{"id": i, "y_true": j, "y_pred": k} for i, j, k in zip(id_list, y_true_list[0], y_pred_list[0])]
        metrics["train_loss"].append(train_loss)
        metrics["test_loss"].append(test_loss)
        metrics["test_p_corr"].append(test_p_corr)

        print(f'Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f} P-CORR: {test_p_corr: .4f}')

    if args.net_outdir:
        torch.save({
            "args": args,
            "metrics": metrics,
            "model_state_dict": net.state_dict(),
        }, os.path.join(args.net_outdir, f'model.pth'))

        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(os.path.join(args.net_outdir, 'predictions.csv'), index=False, encoding="utf-8",)

    return net


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='Train a neural network on tree-structured data.')
    parser.add_argument('--train_file', type=str, required=True, help='Input file with training data.')
    parser.add_argument('--test_file', type=str, required=True, help='Input file with training data.')
    parser.add_argument('--scaler_file', type=str, required=True, help='Scaler file for learned vectors.')

    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--max_seq_length', type=int, default=384)
    parser.add_argument('--vocab_size', type=int, default=221)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--num_transformer_block', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epochs', type=float, default=1)

    parser.add_argument('--net_outdir', type=str, required=True, help='Output file for the neural network model.')

    args = parser.parse_args()
    if args.net_outdir:
        os.makedirs(args.net_outdir, exist_ok=True)

    net = training(args)
