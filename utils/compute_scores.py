import math

import torch.nn as nn
import torch
import sklearn.metrics as metrics
import torch.nn.functional as F


def get_metrics(y):
    """
        Computes how accurately model learns correct matching of object with the caption in terms of accuracy

        Args:
            y(N,2): Tensor(cpu). the incongruity score of negataive class, positive class.

        Returns:
            predict_label (list): predict results
    """
    predict_label = (y[:,0]<y[:,1]).clone().detach().long().numpy().tolist()
    return predict_label

def get_four_metrics(labels, predicted_labels):
    confusion = metrics.confusion_matrix(labels, predicted_labels)
    total = confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1]
    acc = (confusion[0][0] + confusion[1][1])/total
    # about sarcasm
    recall = confusion[1][1]/(confusion[1][1]+confusion[1][0])
    precision = confusion[1][1]/(confusion[1][1]+confusion[0][1])
    f1 = 2*recall*precision/(recall+precision)
    return acc,recall,precision,f1


def L2_Norm(a1, a2):
    """

    Args:
        a1: the alignment distribution of caption1 and
        a2: the alignment distribution of caption2 and

    Returns:

    """
    a1 = a1/math.sqrt(a1.dot(a1))
    a2 = a2/math.sqrt(a2.dot(a2))
    return a1, a2


def L2_norm(X, dim, eps=1e-8):
    norm = torch.sum(torch.pow(X,2),dim=dim,keepdim=True)+eps
    X = torch.div(X, norm)
    return X

def cosine_distance(x1, x2, eps=1e-8):
    # x1 (N,L1,D) x2(N,L2,D)
    # (N,L1,L2)
    w12 = torch.bmm(x1, x2.permute(0,2,1))
    # (N,L1,1)
    w1 = torch.norm(x1,2,dim=2).unsqueeze(2)
    # (N,L2,1)
    w2 = torch.norm(x2,2,dim=2).unsqueeze(2)
    distance = 1 - w12/(torch.bmm(w1, w2.permute(0,2,1))).clamp(min=eps).squeeze()
    return distance



