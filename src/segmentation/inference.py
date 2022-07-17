import argparse
import torch
import cv2, os
import numpy as np
import segmentation_models_pytorch as smp

from dataset import *

SMOOTH = 1e-6


def inference(args, model, device, test_loader, directory):
    os.makedirs(directory, exist_ok=True)
    model.eval()
    # criterion = nn.CrossEntropyLoss()
    for batch_idx, (rgb, data) in enumerate(test_loader):
        # print(batch_idx, 'batch')
        data = data.to(device)
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
        np.save(os.path.join(directory, '{}.npy'.format(batch_idx)), output)

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['roof', 'water'], required=True)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--test-batch-size', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        test_kwargs.update(cuda_kwargs)
    
    for year in range(2016, 2022):
        if args.task == 'water':
            test_dataset = InfDataset_water('data/{}_single_500/'.format(year))
            args.num_classes = 2
        else:
            # test_dataset = InfDataset('data/{}_single_200/'.format(year))
            test_dataset = InfDataset('../cluster_detection/')
            args.num_classes = 3

        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=args.num_classes,       # model output channels (number of classes in your dataset)
        ).to(device)

        if args.task == 'water':
            model.load_state_dict(torch.load('ckpt_883.pt', map_location=torch.device('cpu')))
            inference(args, model, device, test_loader, 'inference_water_{}/'.format(year))
        else:
            model.load_state_dict(torch.load('ckpt_830827.pt', map_location=torch.device('cpu')))
            inference(args, model, device, test_loader, 'inference_roof_{}/'.format(year))


if __name__ == '__main__':
    main()
