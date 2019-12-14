import argparse
from torch.utils.data import DataLoader

from utils import Config
from trainner import Trainner
from data import get_train_val_test_dataset


def main(opt):
    trainner = Trainner(opt)
    train_dataset, validate_dataset, test_dataset = get_train_val_test_dataset(opt)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=opt.batch_size,
                                  pin_memory=opt.pin_memory,
                                  num_workers=opt.num_workers)
    validate_dataloader = DataLoader(validate_dataset,
                                     batch_size=opt.batch_size,
                                     pin_memory=opt.pin_memory,
                                     num_workers=opt.num_workers)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 pin_memory=opt.pin_memory,
                                 num_workers=opt.num_workers)

    trainner.set_test_data(test_dataloader)
    trainner.train(train_dataloader, validate_dataloader)
    print('Finished.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Digit Recognizer")
    parser.add_argument('--cfg', type=str, help='configure file', default='cfg.yaml')
    arg = parser.parse_args()

    opt = Config(arg.cfg)
    main(opt)
