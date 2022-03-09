import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import cv2, os
import numpy as np

from dataset import TrainDataset, TestDataset, TestDataset2
from model import Net



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        print(output.shape)
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
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, mean_loss))


def test(args, model, device, test_loader, epoch):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for batch_idx, (_, data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        losses.append(loss.item())
    mean_loss = torch.tensor(losses).mean().item()
    print('Test  Epoch: {}\tLoss: {:.6f}\n'.format(epoch, mean_loss))


def visualize(args, model, device, test_loader, directory):
    model.eval()
    criterion = nn.CrossEntropyLoss()
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
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--save-model', action='store_true', default=True)
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
    
    train_dataset = TrainDataset()
    test_dataset = TestDataset2()
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # test(args, model, device, test_loader, epoch)
        # visualize(args, model, device, test_loader, 'vis/')
        # scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), 'ckpt.pt')


if __name__ == '__main__':
    main()
