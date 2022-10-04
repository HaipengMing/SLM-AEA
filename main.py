import pickle
import torch
import argparse
from apex import amp
from torchvision import transforms
from torch.utils.data import DataLoader
import tools
from fer_dataset import FerDatasetTrain, FerDatasetVal
from models.ResNet18 import ResNet18, ResNet18Aea
from models.slm import slm_one_epoch, valid_one_epoch, baseline_one_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickled_train', type=str, default='./datasets/RAF-DB_train', help='Train data.')
    parser.add_argument('--pickled_val', type=str, default='./datasets/RAF-DB_test', help='Validation data.')
    parser.add_argument('--pretrained', type=str, default=None, help='Pretrained parameters.')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes of expressions.')
    parser.add_argument('--alpha', type=float, default=1.6,
                        help='Alpha determines the slope and the ramp function’s initial value')
    parser.add_argument('--beta', type=int, default=7,
                        help='Beta is the epoch that the network starts to mine soft labels')
    parser.add_argument('--lam', type=float, default=200,
                        help='Learning rate for updating soft labels.')
    parser.add_argument('--p', type=float, default=0.97,
                        help='The confidence level of the original annotations.')
    parser.add_argument('--k', type=float, default=5,
                        help='The initialization parameters determine the data scale before softmax.')
    parser.add_argument('--batch_size', type=int, default=72, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--lr_max', type=float, default=0.001, help='The initial learning rate.')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='The minimum learning rate to reduce to.')
    parser.add_argument('--weight_decay', default=0.9, type=float, help='Weight decay for Adam.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    return parser.parse_args()


def train():
    args = parse_args()
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    with open(args.pickled_train, 'rb') as f1:
        img_label = pickle.load(f1)
    with open(args.pickled_val, 'rb') as f2:
        val_img_label = pickle.load(f2)

    train_set = FerDatasetTrain(img_label,
                                num_classes=args.num_classes,
                                p=args.p,
                                k=args.k,
                                transform=train_transforms)
    val_set = FerDatasetVal(val_img_label,
                            transform=val_transforms)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              shuffle=True,
                              pin_memory=True
                              )
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            shuffle=False,
                            pin_memory=True
                            )

    hyperparameter = [args.alpha, args.beta, args.lam]
    net = ResNet18Aea(num_classes=args.num_classes).cuda()
    if args.pretrained is not None:
        param_dict = torch.load(args.pretrained, map_location=lambda storage, loc: storage.cpu())
        net.load_state_dict(param_dict)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_max)
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1", verbosity=0)
    scheduler = tools.CosLr(args.epochs * len(train_loader), args.lr_max, args.lr_min)

    epochs = args.epochs
    for epoch in range(epochs):
        acc, loss = slm_one_epoch(epoch, train_loader, net, optimizer, scheduler, train_set, hyperparameter)
        # acc, loss = baseline_one_epoch(epoch, train_loader, net, optimizer, scheduler)
        log = "Train:Epoch[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.4f}".format(epoch + 1, epochs, loss, acc)
        print(log)
        v_acc, v_loss = valid_one_epoch(val_loader, net)
        log = "Valid:Epoch[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.4f}".format(epoch + 1, epochs, v_loss, v_acc)
        print(log)


if __name__ == "__main__":
    train()
