import data
import config
import os
import model
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import warnings
import numpy as np
from utils import write, pad_targets, accuracy_whole_list
import numpy as np
import random

warnings.simplefilter("ignore")

opt = config.parse()
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(opt.seed)
np.random.seed(opt. seed)

def train(opt, train_iter, valid_iter, device):
    net = model.CodeFlow(opt).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)
    write("Start training...", opt.output)
    for epoch in range(opt.num_epoch):
        net.train()
        total_loss, total_accuracy = 0, 0
        total_train = 0
        for batch in tqdm(train_iter):
            x, edges, target = batch.nodes, (batch.forward, batch.backward), batch.target.float()

            if isinstance(x, tuple):
                pred = net(x[0], edges, x[1], x[2])
            else:
                pred = net(x, edges)
            pred = pred.squeeze()
            
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = (pred > opt.beta).float()
            accuracy = accuracy_whole_list(target.cpu().numpy(), pred.cpu().numpy(), x[1].cpu().numpy())
            total_train += target.shape[0]
            total_accuracy += accuracy
        avg_loss = total_loss / len(train_iter)
        avg_accuracy = total_accuracy / total_train

        net.eval()
        eval_loss, eval_accuracy, eval_error_accuracy = 0, 0, 0
        y_true, y_pred = [], []
        total_test = 0
        total_local = 0
        total_detect = 0
        locate_bug = 0
        detect_true = 0
        with torch.no_grad():
            for batch in valid_iter:
                x, edges, target = batch.nodes, (batch.forward, batch.backward), batch.target.float()
                num_nodes = x[1]
                if isinstance(x, tuple):
                    pred = net(x[0], edges, x[1], x[2])
                else:
                    pred = net(x, edges)
                pred = pred.squeeze()
            
                loss = criterion(pred, target)
                eval_loss += loss.item()
                pred = (pred > opt.beta).float()
                if opt.runtime_detection:
                    for i in range(len(x[1])):
                        total_detect += 1
                        if pred[i][x[1][i]-1] == target[i][x[1][i]-1]:
                            detect_true += 1
                if opt.bug_localization:
                    for i in range(len(x[1])):
                        target_list = []
                        pred_list = []
                        num_nodes_list = []
                        if target[i][x[1][i]-1] == 1:
                            continue
                        total_local += 1
                        mask_pred = pred[i] == 1
                        indices_pred = torch.nonzero(mask_pred).flatten()
                        farthest_pred = indices_pred.max().item()

                        mask_target = target[i] == 1                         
                            
                        indices_target = torch.nonzero(mask_target).flatten()
                        farthest_target = indices_target.max().item()
                        if farthest_pred == farthest_target:
                            locate_bug += 1
                        target_list.append(target[i].cpu().numpy())
                        pred_list.append(pred[i].cpu().numpy())
                        num_nodes_list.append(num_nodes[i].cpu().numpy())
                    error_accuracy = accuracy_whole_list(target_list, pred_list, num_nodes_list)
                    eval_error_accuracy += error_accuracy
                accuracy = accuracy_whole_list(target.cpu().numpy(), pred.cpu().numpy(), num_nodes.cpu().numpy())
                eval_accuracy += accuracy
                total_test += target.shape[0]
                # append target to y_true and pred to y_pred base on the number of node in num_nodes
                for i in range(len(num_nodes)):
                    y_true.append(target[i, :num_nodes[i]].cpu().numpy())
                    y_pred.append(pred[i, :num_nodes[i]].cpu().numpy())
        avg_eval_loss = eval_loss / len(valid_iter)
        avg_eval_accuracy = eval_accuracy / total_test
        # concatenate all the target and prediction
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        write(f"Epoch {epoch + 1}/{opt.num_epoch}", opt.output)
        write(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}", opt.output)
        write(f"Validation Loss: {avg_eval_loss:.4f}, Validation Accuracy: {avg_eval_accuracy:.4f}", opt.output)
        if opt.runtime_detection:
            detect_acc = (detect_true / total_detect)*100
            write(f"Runtime Error Detection: {detect_acc:.4f}", opt.output)
        if opt.bug_localization:
            locate_acc = (locate_bug/total_local)*100
            write(f"BUG Localization: {locate_acc:.4f}", opt.output)
        write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {fscore:.4f}", opt.output)
        write("_________________________________________________", opt.output)

        if (epoch+1) % 10 == 0:
            os.makedirs(f'checkpoints/checkpoints_{opt.checkpoint}', exist_ok=True)
            torch.save(net.state_dict(), f"checkpoints/checkpoints_{opt.checkpoint}/epoch-{epoch + 1}.pt")

    return net

def main():
    if opt.checkpoint == None:
        files = os.listdir("checkpoints")
        opt.checkpoint = len(files)+1
    if opt.name_exp == None:
        opt.output = f'{opt.output}/checkpoint_{opt.checkpoint}_{opt.seed}'
    else:
        opt.output = f'{opt.output}/checkpoint_{opt.checkpoint}_{opt.seed}_{opt.name_exp}'
    print(opt.output)
    os.makedirs(os.path.dirname(opt.output), exist_ok=True)
    open(opt.output, 'w').close()
    if opt.cuda_num == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{opt.cuda_num}" if torch.cuda.is_available() else "cpu")
    if opt.seed != None:
        random.seed(opt.seed)
    print(f"Using device: {device}")
    train_iter, test_iter = data.get_iterators(opt, device)
    train(opt, train_iter, test_iter, device)

if __name__ == "__main__":
    main()
