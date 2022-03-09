import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import cv2, os
import numpy as np
import segmentation_models_pytorch as smp

from dataset import *
from model import Net
from criterion import iou_pytorch, iou_numpy, FocalLoss

SMOOTH = 1e-6

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     if args.dry_run:
        #         break
    mean_loss = torch.tensor(losses).mean().item()
    print('\nTrain Epoch: {}\tLoss: {:.6f}'.format(epoch, mean_loss))


def test(args, model, device, test_loader, epoch):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    total_intersection, total_union = [], []
    for batch_idx, (_, data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        losses.append(loss.item())
        output_bin = output.argmax(dim=1)
        intersection, union = iou_pytorch(output_bin, target, args.num_classes)
        total_intersection.append(intersection)
        total_union.append(union)
    total_intersection = torch.tensor(total_intersection).mean(dim=0)
    total_union = torch.tensor(total_union).mean(dim=0)
    mean_iou = (total_intersection + SMOOTH) / (total_union + SMOOTH)
    mean_iou = mean_iou.tolist()
    mean_loss = torch.tensor(losses).mean().item()
    print('Test  Epoch: {}\tLoss: {:.6f}\tIoU: {}'.format(epoch, mean_loss, str(mean_iou)))
    return mean_loss, mean_iou


def visualize(args, model, device, test_loader, directory):
    os.makedirs(directory, exist_ok=True)
    model.eval()
    # criterion = nn.CrossEntropyLoss()
    for batch_idx, (rgb, data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        # print(data.shape)
        output = model(data)
        # print(output.shape)
        # assert(output.shape[0] == 1, 'batch size must be 1')
        output = output.squeeze(0).argmax(0).detach().cpu().numpy()
        rgb = rgb.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        mask = np.zeros_like(rgb)
        mask[output == 0] = np.array([0, 0, 0])
        mask[output == 1] = np.array([255, 0, 0])
        mask[output == 2] = np.array([0, 255, 0])
        rgb = rgb * 255 * 0.7 + mask * 0.3
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(directory, '{}.jpg'.format(batch_idx)), rgb)


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['roof', 'water'], required=True)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--save-model', action='store_true', default=True)
    parser.add_argument('--visualize-only', action='store_true')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    if args.task == 'water':
        train_dataset = TrainDataset_water()
        test_dataset = TestDataset_water()
        args.num_classes = 2
    else:
        train_dataset = TrainDataset()
        test_dataset = TestDataset()
        args.num_classes = 3

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=args.num_classes,       # model output channels (number of classes in your dataset)
    ).to(device)

    if args.visualize_only:
        if args.task == 'water':
            model.load_state_dict(torch.load('ckpt_883.pt'))
            visualize(args, model, device, test_loader, 'vis_unet_water_883/')
        else:
            model.load_state_dict(torch.load('ckpt_830827.pt'))
            visualize(args, model, device, test_loader, 'vis_unet_roof_830827/')

    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        best_iou = 0
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            mean_loss, mean_iou = test(args, model, device, test_loader, epoch)
            mean_iou = np.array(mean_iou).mean()
            if mean_iou > best_iou:
                print('updating best model...')
                best_iou = mean_iou
                visualize(args, model, device, test_loader, 'vis_unet/')
                if args.save_model:
                    torch.save(model.state_dict(), 'ckpt.pt')
            # scheduler.step()


if __name__ == '__main__':
    main()
