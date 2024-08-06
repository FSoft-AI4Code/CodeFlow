import torch
from sklearn.metrics import precision_recall_fscore_support
import warnings
import numpy as np

warnings.simplefilter("ignore")

def write(content, file):
    with open(file, 'a') as f:
        f.write(content + '\n')
        
def pad_targets(target, max_node):
    batch_size, current_max_node = target.shape
    padded_target = torch.zeros(batch_size, max_node, device=target.device)
    padded_target[:, :current_max_node] = target
    return padded_target

def calculate_metrics(y_true, y_pred):
    precisions, recalls, fscores = [], [], []
    for i in range(y_true.shape[1]):
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true[:, i], y_pred[:, i], average='binary')
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls), sum(fscores) / len(fscores)

# def accuracy_whole_list(y_true, y_pred):
#     correct = (y_true == y_pred).all(axis=1).sum().item()
#     return correct / y_true.shape[0]

def accuracy_whole_list(y_true, y_pred, lengths):
    correct = 0
    total = 0
    for i in range(len(lengths)):
        length = lengths[i]
        if np.array_equal(y_true[i][:length], y_pred[i][:length]):
            correct += 1
    return correct