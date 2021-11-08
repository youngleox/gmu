import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

def soft_target(pred, gold, smoothing=0.1):
    #print(gold)
    target = gold.long()
    prob =  1 - (gold - target)
    n_class = pred.size(1)
    #print('target: ', target[0])
    one_hot = (torch.ones_like(pred) * (1 - prob.unsqueeze(1)) / (n_class - 1)).float()
    #print('prob: ', prob[0])
    #print('one hot before: ', one_hot[0,:])
    one_hot.scatter_(dim=1, index=target.unsqueeze(1), src=prob.unsqueeze(1).float())
    #print('one hot after: ', one_hot[0,:])
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob.float(), target=one_hot.float(), reduction='none').sum(-1)
