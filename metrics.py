import torch


def acc(pred, target):
    # print(pred)
    # print(target)
    correct = torch.sum(pred == target)
    # print(correct)
    return correct.type(torch.float32)/pred.size(0)