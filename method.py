import numpy as np
import networkx as nx
from util import eval_metrics
from random import shuffle
import copy

def random_solve(test_seqs,metrics):
    preds = []
    for seq in test_seqs:
        pred_seq = copy.copy(seq)
        shuffle(pred_seq)
        preds.append(pred_seq)
    return eval_metrics(preds, test_seqs, metrics)

def rank_on_activeness(num_user,train_seqs,test_seqs,metrics):
    scores = [[] for i in range(num_user)]
    for seq in train_seqs:
        seq_length = len(seq)
        score = np.arange(seq_length - 1, -1, -1) / (seq_length - 1)
        for i in range(seq_length):
            scores[seq[i]].append(score[i])
    scores = np.array([sum(item)/len(item) if len(item)>0 else 0.5 for item in scores ],dtype=np.float)
    preds = []
    for seq in test_seqs:
        seq_length = len(seq)
        pred_seq = [(seq[i], scores[seq[i]]) for i in range(seq_length)]
        pred_seq = sorted(pred_seq, key=lambda x: x[1], reverse=True)
        pred_seq = [item[0] for item in pred_seq]
        preds.append(pred_seq)

    return eval_metrics(preds, test_seqs,metrics)

def rank_on_influence(num_user,train_seqs,test_seqs,metrics):
    matrix = np.zeros((num_user, num_user), dtype=np.int)
    for seq in train_seqs:
        seq_length = len(seq)
        for i in range(seq_length - 1):
            for j in range(i + 1, seq_length):
                matrix[seq[i], seq[j]] += 1

    matrix = np.array((matrix - matrix.T) > 0, dtype=int)
    graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)

    scores = 1 - np.array(list(nx.pagerank_numpy(graph).values()), dtype=np.float)
    preds = []
    for seq in test_seqs:
        seq_length = len(seq)
        pred_seq = [(seq[i], scores[seq[i]]) for i in range(seq_length)]
        pred_seq = sorted(pred_seq, key=lambda x: x[1], reverse=True)
        pred_seq = [item[0] for item in pred_seq]
        preds.append(pred_seq)

    return eval_metrics(preds, test_seqs, metrics)


