import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import nn
from resnet import resnet
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import metrics
import optim


class Trainner(object):
    def __init__(self, opt):
        self.opt = opt
        self.model = resnet(opt).to(opt.device)
        self.optimizer = optim.optimizer(opt, self.model.parameters())
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.step_size, gamma=opt.gamma)
        self.criterion = nn.BCEWithLogitsLoss().to(opt.device)
        log_dir = os.path.join(opt.log_dir, opt.runtime_id)

        if os.path.exists(log_dir):
            raise RuntimeError(f'{log_dir} already exists.')
        else:
            try:
                os.makedirs(log_dir)
            except Exception as e:
                raise RuntimeError(str(e))

        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_text('config', str(opt), 0)

        self.epoch = 0
        self.best_acc = 0.993
        self.test_dl = None

    def get_pred_number(self, pred):
        return torch.argmax(pred, dim=1)

    def step(self, data, label):
        self.model.train()

        data = data.to(self.opt.device)
        label = label.to(self.opt.device)

        self.optimizer.zero_grad()
        pred = self.model(data)
        loss = self.criterion(pred, label)
        loss.backward()
        self.optimizer.step()
        return loss.detach(), pred
    
    def train_epoch(self, train_dl):
        iterator = tqdm(train_dl, leave=True, dynamic_ncols=True)
        iter_len = len(iterator)
        for i, (_, data, label) in enumerate(iterator):
            self.global_steps = (self.epoch * iter_len) + i
            iterator.set_description(f'train:[{self.epoch}/{self.opt.epochs}|{self.global_steps}]')
            loss, pred = self.step(data, label)

            if self.global_steps % self.opt.sum_freq == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_steps)
                image = make_grid(data[:self.opt.disp_images, :, :, :].clone().cpu().data,
                                  nrow=self.opt.disp_images, padding=2)
                self.writer.add_image('train/input', image, self.global_steps)
    
    def train(self, train_dl, validate_dl):
        while self.epoch < self.opt.epochs:
            self.train_epoch(train_dl)
            self.validate(validate_dl)
            self.epoch += 1
            # self.scheduler.step()

    def set_test_data(self, test_dl):
        self.test_dl = test_dl

    def test(self):
        test_dl = self.test_dl
        if test_dl is None:
            raise RuntimeError('test dataload is None.')
        iterator = tqdm(test_dl, leave=True, dynamic_ncols=True)
        itre_len = len(iterator)
        result = []

        for i, (index, data, _) in enumerate(iterator):
            iterator.set_description(f'test:')
            _, pred = self.validate_step(data, None)
            target = self.get_pred_number(pred)
            result.append([index + 1, target.item()])
        self.write_result(np.array(result))
        
    def validate(self, validate_dl):
        iterator = tqdm(validate_dl, leave=True, dynamic_ncols=True)
        iter_len = len(iterator)
        errs = []
        accs = []

        for i, (_, data, label) in enumerate(iterator):
            iterator.set_description(f'validate:[{self.epoch}/{self.opt.epochs}|{self.global_steps}]')
            err, pred = self.validate_step(data, label)

            pred = self.get_pred_number(pred)
            label = self.get_pred_number(label.to(self.opt.device))
            acc = metrics.acc(pred, label)

            accs.append(acc)
            errs.append(err)
        
        curr_acc = torch.tensor(accs).mean().item()
        self.writer.add_scalar('validate/error', torch.tensor(errs).mean().item(), self.global_steps)
        self.writer.add_scalar('validate/acc', curr_acc, self.global_steps)
        if curr_acc > self.best_acc:
            self.best_acc = curr_acc
            self.test()

    def validate_step(self, data, label=None):
        self.model.eval()

        data = data.to(self.opt.device)
        if label is not None:
            label = label.to(self.opt.device)

        pred = self.model(data)

        if label is not None:
            err = self.criterion(pred, label).detach()
        else:
            err = 0

        return err, pred

    def write_result(self, data):
        df = pd.DataFrame({'ImageId': data[:, 0],
                           'Label': data[:, 1]})
        df.to_csv(self.opt.out_file, index=False)

