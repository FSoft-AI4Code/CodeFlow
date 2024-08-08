import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import write

class CodeFlow(nn.Module):
    def __init__(self, opt):
        super(CodeFlow, self).__init__()
        self.max_node = opt.max_node
        self.hidden_dim = opt.hidden_dim
        self.embedding = nn.Embedding(opt.vocab_size+2, opt.hidden_dim, padding_idx=1)
        self.node_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.gate = Gate(self.hidden_dim, self.hidden_dim)
        self.back_gate = Gate(self.hidden_dim, self.hidden_dim)
        self.concat = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_output = nn.Linear(self.hidden_dim, 1)  # Adjusted this layer to match the hidden dimensions
        self.opt = opt

    # @profile
    def forward(self, x, edges, node_lens=None, token_lens=None, target=None):
        f_edges, b_edges = edges
        batch_size, num_node, num_token = x.size(0), x.size(1), x.size(2)
        
        # token_ids  [bs,num_node,num_token]
        # x:                ([bs, 34, 12],        [bs],      [bs, 34])
        #                                       num_node  lengths of node (num_token)
        # f_edges, b_edges:  ([bs, 38, 2], [bs, 6, 2]) -> max_edge_forward, max_edge_back
        # target: [bs, 34]
                            
        x = self.embedding(x)  # [B, N, L, H]
        if self.opt.extra_aggregate:
            neigbors = [{} for _ in range(len(f_edges))]
            for i in range(len(f_edges)):
                for (start, end) in f_edges[i]:
                    start = start.item()
                    end = end.item()
                    if start == 1 and end == 1:
                        continue
                    if start not in neigbors[i]:
                        neigbors[i][start] = [end]
                    else:
                        neigbors[i][start].append(end)
                    if end not in neigbors[i]:
                        neigbors[i][end] = [start]
                    else:
                        neigbors[i][end].append(start)
            for i in range(len(b_edges)):
                for (start, end) in b_edges[i]:
                    start = start.item()
                    end = end.item()
                    if start == 1 and end == 1:
                        continue
                    if start not in neigbors[i]:
                        neigbors[i][start] = [end]
                    else:
                        neigbors[i][start].append(end)
                    if end not in neigbors[i]:
                        neigbors[i][end] = [start]
                    else:
                        neigbors[i][end].append(start)
            max_node = max(node_lens)
            matrix = torch.zeros((batch_size, max_node, max_node), dtype=torch.float, device=x.device)
            for i in range(batch_size):
                if self.opt.delete_redundant_node:
                    for node in neigbors[i]:
                        num_neigbors = len(neigbors[i][node])
                        for neighbor in neigbors[i][node]:
                            matrix[i, node-1, neighbor-1] = (1-self.opt.alpha)/num_neigbors
                        matrix[i, node-1, node-1] = self.opt.alpha
                else:
                    for node in range(max_node):
                        if node in neigbors[i].keys():
                            num_neigbors = len(neigbors[i][node])
                            for neighbor in neigbors[i][node]:
                                matrix[i, node-1, neighbor-1] = (1-self.opt.alpha)/num_neigbors
                            matrix[i, node-1, node-1] = self.opt.alpha
                        else:
                            matrix[i, node-1, node-1] = 1

        #! Node LSTM embedding, https://www.readcube.com/library/1771e2fb-bec1-4bc4-90b3-04c8786fe9dd:fd440d39-f13e-430c-b768-751878616cda, 2nd figure, Node Embedding part
        if token_lens is not None:
            x = x.view(batch_size*num_node, num_token, -1)
            h_n = torch.zeros((2, batch_size*num_node, self.hidden_dim//2)).to(x.device)
            c_n = torch.zeros((2, batch_size*num_node, self.hidden_dim//2)).to(x.device)
            x, _ = self.node_lstm(x, (h_n, c_n))  # [B*N, L, H]
            x = x.view(batch_size, num_node, num_token, -1)
            x = self.average_pooling(x, token_lens)
        else:
            x = torch.mean(x, dim=2)  # [B, N, H]
        
        # ! Initialize hidden states to be zeros
        h_f = torch.zeros(x.size()).to(x.device)
        c_f = torch.zeros(x.size()).to(x.device)

        # ! Forward pass: including forward egde + backward edge, 1->K
        ori_f_matrix = self.convert_to_matrix(batch_size, num_node, f_edges)
        running_f_matrix = ori_f_matrix.clone()
        for i in range(num_node):
            f_i = running_f_matrix[:, i, :].unsqueeze(1)
            f_i = f_i.clone()
            x_cur = x[:, i, :].squeeze(1) # [B, hidden_dim]
            h_last, c_last = f_i.bmm(h_f), f_i.bmm(c_f) # h = [B, max_node, H]
            # h_last = [B, 1, H]
            # Stopping to check if the node is binary
            # [B, 1, max_node] * [B, max_node, hidden_dim] = [B, 1, hidden_dim]
            # h_last, c_last = [B, 1, hidden_dim]
            h_i, c_i = self.gate(x_cur, h_last.squeeze(1), c_last.squeeze(1))
            h_f[:, i, :], c_f[:, i, :] = h_i, c_i
            # make the f_matrix, the next nodes j, which connect to i->j. Change their jth row at ith entry
            h_i, c_i = h_i.squeeze(1), c_i.squeeze(1)
            # for sample_id in range(batch_size):
            #     next_node_ids = []
            #     for j in range(num_node):
            #         if running_f_matrix[sample_id, j, i] == 1:
            #             next_node_ids.append(j)
                
            #     if len(next_node_ids) > 2:
            #         print(sample_id)
            #         print(torch.sum(running_f_matrix, dim=1))
            #         # raise ValueError(f"Node {i+1} in sample_id: {sample_id} has more than 2 outward edges")
            #     if len(next_node_ids) == 2:
            #         if h_i[sample_id].sum() >= 0:
            #             running_f_matrix[sample_id, next_node_ids[0], i] = 0
            #         else:
            #             running_f_matrix[sample_id, next_node_ids[1], i] = 0
            

        b_matrix = self.convert_to_matrix(batch_size, num_node, b_edges)
        for j in range(num_node):
            b_j = b_matrix[:, j, :].unsqueeze(1)
            h_temp = b_j.bmm(h_f)
            h_f[:, j, :] += h_temp.squeeze(1)

        # # ! Initialize hidden states to be zeros
        # h_b = torch.zeros(x.size()).to(x.device)
        # c_b = torch.zeros(x.size()).to(x.device)

        # # # ! Backward pass: transpose b_matrix, f_matrix, including forward egde + backward edge, K->1
        # b_matrix = self.convert_to_matrix(batch_size, num_node, f_edges)
        # b_matrix = b_matrix.transpose(1, 2)
        # for i in reversed(range(num_node)):
        #     x_cur = x[:, i, :].squeeze(1)
        #     b_i = b_matrix[:, i, :].unsqueeze(1)
        #     h_hat, c_hat = b_i.bmm(h_b), b_i.bmm(c_b)
        #     h_b[:, i, :], c_b[:, i, :] = self.back_gate(x_cur, h_hat.squeeze(), c_hat.squeeze())

        # f_matrix = self.convert_to_matrix(batch_size, num_node, b_edges)
        # f_matrix = f_matrix.transpose(1, 2)
        # for j in range(num_node):
        #     f_j = f_matrix[:, j, :].unsqueeze(1)
        #     h_temp = f_j.bmm(h_b)
        #     h_b[:, j, :] += h_temp.squeeze(1)

        # ------------Prediction stage --------------#
        
        # h = torch.cat([h_f, h_b], dim=2)
        # output = torch.mean(h, dim=1) # take the mean over the nodes within a batch -> [B, H]
        # h = [B, max_node, H] -> each node is feeded into the fc_output
        
        # B, max_node, H -> B, max_node
        output = torch.sigmoid(self.fc_output(h_f)) # 
        if self.opt.extra_aggregate:
            output = torch.bmm(matrix, output)
        return output

    @staticmethod
    def average_pooling(data, input_lens):
        B, N, T, H = data.size()
        idx = torch.arange(T, device=data.device).unsqueeze(0).expand(B, N, -1)
        idx = idx < input_lens.unsqueeze(2)
        idx = idx.unsqueeze(3).expand(-1, -1, -1, H)
        ret = (data.float() * idx.float()).sum(2) / (input_lens.unsqueeze(2).float()+10**-32)
        return ret

    @staticmethod
    def convert_to_matrix(batch_size, max_num, m):
        matrix = torch.zeros((batch_size, max_num, max_num), dtype=torch.float, device=m.device)
        m -= 1
        b_select = torch.arange(batch_size).unsqueeze(1).expand(batch_size, m.size(1)).contiguous().view(-1)
        matrix[b_select, m[:, :, 1].contiguous().view(-1), m[:, :, 0].contiguous().view(-1)] = 1
        matrix[:, 0, 0] = 0
        return matrix


class Gate(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(Gate, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ax = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.ah = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def forward(self, inputs, last_h, pred_c):
        iou = self.ax(inputs) + self.ah(last_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(self.fh(last_h) + self.fx(inputs))
        fc = torch.mul(f, pred_c)
        c = torch.mul(i, u) + fc
        h = torch.mul(o, torch.tanh(c))
        return h, c