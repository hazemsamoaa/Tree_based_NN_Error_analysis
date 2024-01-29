import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from metrics import pearson_corr_v2 as pearson_corr
from models.code2vec.net import Code2vecNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def batch_samples(data, batch_size):
    """Batch samples from a generator"""
    representations, labels = [], []
    samples = 0
    for x in data:
        representations.append(x["representation"])
        labels.append(x["y"])
        samples += 1
        if samples >= batch_size:
            yield _pad_batch(representations, labels)
            representations, labels = [], []
            samples = 0


def _pad_batch(representations, labels):
    if not representations:
        return [], []

    max_rows = max(arr.shape[0] for arr in representations)
    max_cols = representations[0].shape[1]
    padded_representations = [
        np.expand_dims(np.vstack([arr, np.zeros((max_rows - arr.shape[0], max_cols))]), axis=0)
        for arr in representations
    ]

    labels = np.array(labels)

    return np.concatenate(padded_representations, axis=0), labels



def trainer(model, train, test, y_scaler, device, lr=1e-3, batch_size=8, epochs=1, checkpoint=1000, output_dir=None):
    output_dir = output_dir if output_dir and len(output_dir) > 0 else None
    # Cross Entropy loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize step count
    total_samples = len(train)
    step_count = 0

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        # P-CORR
        y_pred_list = []
        y_true_list = []

        for i, batch in enumerate(batch_samples(train, batch_size=batch_size)):
            batch_code_vectors, batch_labels = batch

            batch_code_vectors = torch.from_numpy(batch_code_vectors).float().to(device)
            batch_labels = y_scaler.transform(batch_labels.reshape(-1, 1))
            batch_labels = torch.tensor(batch_labels).float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch_code_vectors)
            
            y_pred = logits.cpu().detach().flatten().numpy().tolist()
            y_true = batch_labels.cpu().detach().flatten().numpy().tolist()
            y_pred_list += y_pred if isinstance(y_pred, list) else [y_pred]
            y_true_list += y_true if isinstance(y_true, list) else [y_true]

            # Compute loss
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update step count
            step_count += 1

            # Checkpoint based on steps
            if checkpoint > 0 and step_count % checkpoint == 0:
                logger.info(f'Epoch [{epoch + 1}/{epochs}], Step [{step_count}], Loss: {total_loss / (i + 1):.4f}')
                # torch.save(net.state_dict(), f'checkpoint_step_{step_count}.pth')

        # Log per-epoch statistics
        y_pred_list = torch.tensor(y_scaler.inverse_transform(np.array(y_pred_list).reshape(1, -1))).squeeze()
        y_true_list = torch.tensor(y_scaler.inverse_transform(np.array(y_true_list).reshape(1, -1))).squeeze()

        p_corr = pearson_corr(y_pred_list, y_true_list)
        train_msg = f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / total_samples:.4f}, P-CORR: {p_corr}\n'
        logger.info(train_msg)

        if output_dir:
            with open(os.path.join(output_dir, 'train_pred.txt'), "w", encoding="utf-8") as f:
                f.write(f"TRUE\tPRED\n")
                for i, j in zip(y_true_list.tolist(), y_pred_list.tolist()):
                    f.write(f"{i}\t{j}\n")

    total_samples = len(test)
    total_loss = 0.0

    # P-CORR
    y_pred_list = []
    y_true_list = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(batch_samples(test, batch_size=batch_size)):
            batch_code_vectors, batch_labels = batch

            batch_code_vectors = torch.from_numpy(batch_code_vectors).float().to(device)
            batch_labels = y_scaler.transform(batch_labels.reshape(-1, 1))
            batch_labels = torch.tensor(batch_labels).float().to(device)

            # Forward pass
            logits = model(batch_code_vectors)
            
            # y_pred_list.append(logits.item())
            # y_true_list.append(batch_labels.item())
            y_pred = logits.cpu().detach().flatten().numpy().tolist()
            y_true = batch_labels.cpu().detach().flatten().numpy().tolist()
            y_pred_list += y_pred if isinstance(y_pred, list) else [y_pred]
            y_true_list += y_true if isinstance(y_true, list) else [y_true]

            # Compute loss
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()

        # Log per-epoch statistics
        y_pred_list = torch.tensor(y_scaler.inverse_transform(np.array(y_pred_list).reshape(1, -1))).squeeze()
        y_true_list = torch.tensor(y_scaler.inverse_transform(np.array(y_true_list).reshape(1, -1))).squeeze()

        p_corr = pearson_corr(y_pred_list, y_true_list)
        eval_msg = f'Loss: {total_loss / total_samples:.4f}, P-CORR: {p_corr}'
        logger.info(eval_msg)

    if output_dir:
        torch.save(model.state_dict(), os.path.join(output_dir, f'model.pth'))
        with open(os.path.join(output_dir, 'output.txt'), "w", encoding="utf-8") as f:
            f.write(f"TRAIN: {train_msg}")
            f.write(f" EVAL: {eval_msg}")

        with open(os.path.join(output_dir, 'eval_pred.txt'), "w", encoding="utf-8") as f:
            f.write(f"TRUE\tPRED\n")
            for i, j in zip(y_true_list.tolist(), y_pred_list.tolist()):
                f.write(f"{i}\t{j}\n")

    return model
