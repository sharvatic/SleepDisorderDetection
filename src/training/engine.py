import torch
import numpy as np

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """
    Standard training loop for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        criterion (callable): Loss function.
        optimizer (Optimizer): PyTorch optimizer.
        device (torch.device): CPU or CUDA.
        scaler (GradScaler, optional): For mixed precision scaling.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for tensors, disorder_labels, _ in loader:
        tensors         = tensors.to(device)
        disorder_labels = disorder_labels.to(device)

        optimizer.zero_grad()
        
        # Automatic Mixed Precision
        if scaler is not None and scaler.is_enabled():
            with torch.cuda.amp.autocast():
                outputs = model(tensors) 
                loss    = criterion(outputs, disorder_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(tensors) 
            loss    = criterion(outputs, disorder_labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * tensors.size(0)
        predicted   = outputs.argmax(dim=1)
        correct    += (predicted == disorder_labels).sum().item()
        total      += tensors.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, use_amp=False):
    """
    Validation/Test evaluation loop. 
    Does not update weights.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Data loader.
        criterion (callable): Loss function.
        device (torch.device): CPU or CUDA.
        use_amp (bool): Whether to use half precision during inference.

    Returns:
        tuple: (average_loss, accuracy, predictions, true_labels)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for tensors, disorder_labels, _ in loader:
            tensors         = tensors.to(device)
            disorder_labels = disorder_labels.to(device)

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs   = model(tensors)
                    loss      = criterion(outputs, disorder_labels)
            else:
                outputs   = model(tensors)
                loss      = criterion(outputs, disorder_labels)

            total_loss += loss.item() * tensors.size(0)
            predicted   = outputs.argmax(dim=1)
            correct    += (predicted == disorder_labels).sum().item()
            total      += tensors.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(disorder_labels.cpu().numpy())

    avg_loss = total_loss / total
    avg_acc  = correct / total
    
    return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels)
