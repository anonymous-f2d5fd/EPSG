import numpy as np
import torch
import random
import torch.nn as nn
import os
from collections import defaultdict
from math import ceil


class SequenceBatchNLLLoss(nn.Module):
    def __init__(self):
        super(SequenceBatchNLLLoss, self).__init__()
        self.eps = 1e-44

    def forward(self, logits, preds):
        logits, preds = (logits[:,:-1,:] + self.eps).log(), preds.unsqueeze(-1)
        logits = - torch.gather(logits,dim=-1,index=preds).squeeze(-1)
        logits = logits.sum() / logits.shape[0]
        return logits


def eval_metric(seq_list0, seq_list1, metric='rouge_l'):
    assert metric in ('rouge_l', 'rouge_s','rouge_2'), 'evaluation metric not implement!'
    assert isinstance(seq_list0,list) and isinstance(seq_list0,list), 'input not list!'
    assert len(seq_list0) == len(seq_list1), 'input list size not match!'
    calculate = {'rouge_l': rouge_l,
                 'rouge_s': rouge_s,
                 'rouge_2': rouge_2}
    size = len(seq_list0)
    res = 0
    for seq0, seq1 in zip(seq_list0, seq_list1):
        res += calculate[metric](seq0,seq1)
    return res/size


def eval_metrics(seq_list0, seq_list1, metrics):
    result = []
    for metric in metrics:
        assert metric in ('rouge_l', 'rouge_s','rouge_2'), 'evaluation metric not implement!'
        result.append(eval_metric(seq_list0,seq_list1,metric))
    return result


def rouge_l(seq0, seq1):
    size = len(seq0)
    dp = np.zeros((size+1, size+1), dtype=np.int)
    for i in range(1,size+1):
        for j in range(1,size+1):
            if seq0[i-1] == seq1[j-1]:
                dp[i,j] = dp[i-1,j-1] +1
            else:
                dp[i,j] = max(dp[i-1,j], dp[i,j-1])

    return dp[size][size] / size


def rouge_s(seq0, seq1):
    set0, set1 = set(), set()
    size = len(seq1)
    for i in range(size - 1):
        for j in range(i + 1, min(i + 3, size)):
            set0.add((seq0[i], seq0[j]))
            set1.add((seq1[i], seq1[j]))
    return len(set0 & set1) / len(set0)


def rouge_2(seq0, seq1):
    set0, set1 = set(), set()
    size = len(seq1)
    for i in range(size - 1):
            set0.add((seq0[i], seq0[i+1]))
            set1.add((seq1[i], seq1[i+1]))
    return len(set0 & set1) / len(set0)

def load_and_split_data(path,device,train_pr=0.7,val_pr=0.1,random_seed=0):
    dataset = torch.load(path).to(device)
    num_event = dataset['num_node_dic']['event']
    sequence = dataset['sequence_dic']

    idx = np.arange(num_event)
    random.seed(random_seed)
    random.shuffle(idx)
    train_idx = idx[:int(num_event * train_pr)]
    val_idx = idx[int(num_event * train_pr):int(num_event * (train_pr + val_pr))]
    test_idx = idx[int(num_event * (train_pr + val_pr)):]
    train = dict([(key, sequence[key]) for key in train_idx])
    val = dict([(key, sequence[key]) for key in val_idx])
    test = dict([(key, sequence[key]) for key in test_idx])
    dataset['sequence_dic'] = None
    return dataset,train,val,test


def sequence_data_load(sequence_dic, max_batch=2000, device='cpu'):
    count_dic = defaultdict(list)
    for k,v in sequence_dic.items():
        count_dic[len(v)].append(k)
    for _, events in count_dic.items():
        length = len(events)
        for i in range(ceil(length/ max_batch)):
            begin = i * max_batch
            end = min((i+1) * max_batch, length)
            event_ids = torch.tensor(events[begin:end], dtype=torch.long).to(device)
            event_sequences = torch.stack([sequence_dic[event_id] for event_id in events[begin:end]],dim=0).to(device)
            yield event_ids,event_sequences

def test_model(model,dataset,data,batch_size,device,methods=('rouge_l', 'rouge_s','rouge_2'),save_path='./data',save_data=False):
    preds = []
    targets = []
    id_list = []
    for ids, seqs in sequence_data_load(data, batch_size, device):
        id_list.extend(list(ids.clone().cpu().numpy()))

        targets.extend(list(seqs.clone().cpu().numpy()))
        raw_feature = dict([(k, v.clone()) for k, v in dataset['x'].items()])
        pred_seqs = model.test(raw_feature, batch_size, dataset['edge_index'], ids, seqs)
        preds.extend(list(pred_seqs.clone().cpu().numpy()))
    if save_data:
        raw_feature = dict([(k, v.clone()) for k, v in dataset['x'].items()])
        embedding= model.get_embedding(raw_feature, dataset['edge_index'])
        res = {'id':id_list,'pred':preds,'true':targets,'embedding':embedding}
        torch.save(res,os.path.join(save_path,'model_data.pt'))


    return eval_metrics(preds,targets,methods)

def train_model(model,opt,loss_func,dataset,data,batch_size,device):
    total_loss = 0
    for ids, seqs in sequence_data_load(data, batch_size, device):

        opt.zero_grad()
        raw_feature = dict([(k, v.clone()) for k, v in dataset['x'].items()])
        logits = model(raw_feature, batch_size, dataset['edge_index'], ids, seqs)
        loss = loss_func(logits, seqs)
        loss.backward()

        opt.step()
        total_loss += loss.item()
    return total_loss


@torch.no_grad()
def eval_model(model,loss_func,dataset,data,batch_size,device):
    eval_loss = 0
    for ids, seqs in sequence_data_load(data, batch_size, device):
        raw_feature = dict([(k, v.clone()) for k, v in dataset['x'].items()])
        logits = model(raw_feature, batch_size, dataset['edge_index'], ids, seqs)
        loss = loss_func(logits, seqs)
        eval_loss += loss.item()
    return eval_loss