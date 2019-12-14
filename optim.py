from torch.optim import Adam, SGD


optims = {
    'Adam': Adam,
    'SGD': SGD
}


def optimizer(opt, parameters):
    optimizer = optims[opt.optimizer]
    return optimizer(parameters, lr=opt.lr)
