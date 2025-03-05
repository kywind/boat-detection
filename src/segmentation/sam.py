import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import cv2, os
import numpy as np
# import segmentation_models_pytorch as smp

from dataset import *
from criterion import iou_pytorch, iou_numpy, FocalLoss

from PIL import Image
from lang_sam import LangSAM

SMOOTH = 1e-6

# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     accs = []
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         # optimizer.zero_grad()
        
#         image_predictor.set_image(np.array(image.convert("RGB")))
#         masks, scores, logits = image_predictor.predict(
#             point_coords=None,
#             point_labels=None,
#             box=input_boxes,
#             multimask_output=False,
#         )

        # acc = criterion(output, target)
        # acc.backward()
        # optimizer.step()
        # accs.append(acc.item())
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tacc: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), acc.item()))
        #     if args.dry_run:
        #         break
    # mean_acc = torch.tensor(accs).mean().item()
    # print('\nTrain Epoch: {}\tacc: {:.6f}'.format(epoch, mean_acc))


def test_roof(args, model, device, test_loader, epoch):
    # model.eval()
    # criterion = nn.CrossEntropyLoss()
    # losses = []
    total_intersection, total_union = [], []
    for batch_idx, (_, data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        # import ipdb; ipdb.set_trace()
        # image_predictor.set_image(np.array(image.convert("RGB")))
        # masks, scores, logits = image_predictor.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=input_boxes,
        #     multimask_output=False,
        # )

        data = data[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        image_pil = Image.fromarray(data.astype(np.uint8))
        image_pil.save(f'log/pred_vis_roof/{batch_idx}.png')
        
        output = torch.zeros((1, 3, image_pil.size[1], image_pil.size[0]), device=target.device)
        output[:, 0] = 0.2
        for ti, text_prompt in enumerate([
            "thatch-roof chicken house.",
            "zinc-roof chicken house.", 
        ]):
            results = model.predict([image_pil], [text_prompt])

            mask = results[0]['masks']
            scores = results[0]['scores']
            # means = np.zeros((mask.shape[0], 2))


            mask_all = np.zeros((mask.shape[1], mask.shape[2]))
            for i in range(mask.shape[0]):
                mask_i = mask[i]
                h_coords, w_coords = np.where(mask_i > 0.5)
                if h_coords.size > 10000:
                    pass
                    # means[i] = [-10000, -10000]
                else:
                    mask_all = np.maximum(mask_all, mask[i] * scores[i])
                    # h_mean = h_coords.mean()
                    # w_mean = w_coords.mean()
                    # means[i] = [h_mean, w_mean]
            
            # mean_h = image_pil.size[1] / 2
            # mean_w = image_pil.size[0] / 2

            # i_best = 0
            # for i in range(1, mask.shape[0]):
            #     if np.linalg.norm(means[i] - [mean_h, mean_w]) < np.linalg.norm(means[i_best] - [mean_h, mean_w]):
            #         i_best = i
            
            # mask_best = mask[i_best]

            # mask_best = (mask_all > 0.5).astype(np.uint8)
            mask_best = mask_all

            mask_best_vis = (mask_best * 255).astype(np.uint8)
            Image.fromarray(mask_best_vis).save(f'log/pred_vis_roof/{batch_idx}_pred_{ti+1}.png')

            mask_best = torch.tensor(mask_best, device=target.device).unsqueeze(0)
            output[:, (ti + 1)] = mask_best

        # import ipdb; ipdb.set_trace()

        # loss = criterion(output, target)
        # losses.append(loss.item())

        # output_vis = output.squeeze(0).permute(1, 2, 0) * 255
        # output_vis = output_vis.detach().cpu().numpy().astype(np.uint8)
        # output_vis = cv2.cvtColor(output_vis, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f'log/pred_vis_roof/{batch_idx}_pred_vis.png', output_vis)

        output_bin = output.argmax(dim=1)

        # output_bin = torch.clamp(output_bin, 0, 1)
        # target = torch.clamp(target, 0, 1)
        # args.num_classes = 2

        output_bin_vis = np.zeros((output_bin.size(1), output_bin.size(2), 3), dtype=np.uint8)
        output_bin_vis[output_bin[0].cpu().numpy() == 0] = [100, 0, 0]
        output_bin_vis[output_bin[0].cpu().numpy() == 1] = [0, 255, 0]
        output_bin_vis[output_bin[0].cpu().numpy() == 2] = [0, 0, 255]
        output_bin_vis = cv2.cvtColor(output_bin_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'log/pred_vis_roof/{batch_idx}_pred_vis.png', output_bin_vis)

        target_vis = np.zeros((target.size(1), target.size(2), 3), dtype=np.uint8)
        target_vis[target[0].cpu().numpy() == 0] = [100, 0, 0]
        target_vis[target[0].cpu().numpy() == 1] = [0, 255, 0]
        target_vis[target[0].cpu().numpy() == 2] = [0, 0, 255]
        target_vis = cv2.cvtColor(target_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'log/pred_vis_roof/{batch_idx}_pred_vis_gt.png', target_vis)

        intersection, union = iou_pytorch(output_bin, target, args.num_classes)
        total_intersection.append(intersection)
        total_union.append(union)

    total_intersection = torch.tensor(total_intersection).mean(dim=0)
    total_union = torch.tensor(total_union).mean(dim=0)
    mean_iou = (total_intersection + SMOOTH) / (total_union + SMOOTH)
    mean_iou = mean_iou.tolist()
    # mean_loss = torch.tensor(losses).mean().item()

    mean_loss = 0.
    print('Test  Epoch: {}\tLoss: {:.6f}\tIoU: {}'.format(epoch, mean_loss, str(mean_iou)))
    return mean_loss, mean_iou



def test_water(args, model, device, test_loader, epoch):
    # model.eval()
    # criterion = nn.CrossEntropyLoss()
    # losses = []
    total_intersection, total_union = [], []
    for batch_idx, (_, data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        # import ipdb; ipdb.set_trace()
        # image_predictor.set_image(np.array(image.convert("RGB")))
        # masks, scores, logits = image_predictor.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=input_boxes,
        #     multimask_output=False,
        # )

        data = data[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        image_pil = Image.fromarray(data.astype(np.uint8))
        image_pil.save(f'log/pred_vis_water/{batch_idx}.png')
        
        output = torch.zeros((1, 3, image_pil.size[1], image_pil.size[0]), device=target.device)
        output[:, 0] = 0.2
        for ti, text_prompt in enumerate([
            "fish pond.",
        ]):
            results = model.predict([image_pil], [text_prompt])

            mask = results[0]['masks']
            scores = results[0]['scores']
            # means = np.zeros((mask.shape[0], 2))


            mask_all = np.zeros((mask.shape[1], mask.shape[2]))
            for i in range(mask.shape[0]):
                mask_i = mask[i]
                h_coords, w_coords = np.where(mask_i > 0.5)
                if h_coords.size > 100000:
                    pass
                else:
                    mask_all = np.maximum(mask_all, mask[i] * scores[i])

            mask_best = mask_all

            mask_best_vis = (mask_best * 255).astype(np.uint8)
            Image.fromarray(mask_best_vis).save(f'log/pred_vis_water/{batch_idx}_pred_{ti+1}.png')

            mask_best = torch.tensor(mask_best, device=target.device).unsqueeze(0)
            output[:, (ti + 1)] = mask_best

        output_bin = output.argmax(dim=1)

        # output_bin = torch.clamp(output_bin, 0, 1)
        # target = torch.clamp(target, 0, 1)
        # args.num_classes = 2

        output_bin_vis = np.zeros((output_bin.size(1), output_bin.size(2), 3), dtype=np.uint8)
        output_bin_vis[output_bin[0].cpu().numpy() == 0] = [100, 0, 0]
        output_bin_vis[output_bin[0].cpu().numpy() == 1] = [0, 255, 0]
        output_bin_vis[output_bin[0].cpu().numpy() == 2] = [0, 0, 255]
        output_bin_vis = cv2.cvtColor(output_bin_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'log/pred_vis_water/{batch_idx}_pred_vis.png', output_bin_vis)

        target_vis = np.zeros((target.size(1), target.size(2), 3), dtype=np.uint8)
        target_vis[target[0].cpu().numpy() == 0] = [100, 0, 0]
        target_vis[target[0].cpu().numpy() == 1] = [0, 255, 0]
        target_vis[target[0].cpu().numpy() == 2] = [0, 0, 255]
        target_vis = cv2.cvtColor(target_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'log/pred_vis_water/{batch_idx}_pred_vis_gt.png', target_vis)

        intersection, union = iou_pytorch(output_bin, target, args.num_classes)
        total_intersection.append(intersection)
        total_union.append(union)

    total_intersection = torch.tensor(total_intersection).mean(dim=0)
    total_union = torch.tensor(total_union).mean(dim=0)
    mean_iou = (total_intersection + SMOOTH) / (total_union + SMOOTH)
    mean_iou = mean_iou.tolist()
    # mean_loss = torch.tensor(losses).mean().item()

    mean_loss = 0.
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
        rgb_vis = rgb * 255 * 0.7 + mask * 0.3
        rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(directory, '{}.jpg'.format(batch_idx)), rgb_vis)

        target = target.squeeze(0).detach().cpu().numpy()
        mask = np.zeros_like(rgb)
        mask[target == 0] = np.array([0, 0, 0])
        mask[target == 1] = np.array([255, 0, 0])
        mask[target == 2] = np.array([0, 255, 0])
        rgb_vis = rgb * 255 * 0.7 + mask * 0.3
        rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(directory, '{}_gt.jpg'.format(batch_idx)), rgb_vis)


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
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--level18', action='store_true')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    args.batch_size = 1
    args.test_batch_size = 1

    # train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        # train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    if args.task == 'water':
        # train_dataset = TrainDataset_water()
        test_dataset = TestDataset_water(im_root='log/semantic_segmentation/water_out_example/JPEGImages',
                                         gt_root='log/semantic_segmentation/water_out_example/SegmentationClass')
        args.num_classes = 2
    else:
        # if args.level18:
        #     # train_dataset = TrainDataset(im_root='data_roof/JPEGImages_Level18')
        #     test_dataset = TestDataset(im_root='data_roof/JPEGImages_Level18')
        # else:
            # train_dataset = TrainDataset(im_root='data_roof/JPEGImages')
        test_dataset = TestDataset(im_root='log/semantic_segmentation/roof_out_example/JPEGImages',
                                   gt_root='log/semantic_segmentation/roof_out_example/SegmentationClass')
        args.num_classes = 3

    # train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # model = smp.Unet(
    #     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=args.num_classes,       # model output channels (number of classes in your dataset)
    # ).to(device)

    # checkpoint = str(root.parent / "weights/sam2/sam2.1_hiera_large.pt")
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # image_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    # # video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    # model = image_predictor

    model = LangSAM()

    # model_id = "IDEA-Research/grounding-dino-tiny"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # processor = AutoProcessor.from_pretrained(model_id)
    # grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    if args.test_only:
        if args.task == 'water':
            model.load_state_dict(torch.load('ckpt_883.pt'))
            mean_loss, mean_iou = test(args, model, device, test_loader, 0)
            if args.visualize: visualize(args, model, device, test_loader, 'vis_unet_water_883/')
        else:
            model.load_state_dict(torch.load('ckpt_830827.pt'))
            mean_loss, mean_iou = test(args, model, device, test_loader, 0)
            if args.visualize: visualize(args, model, device, test_loader, 'vis_unet_roof_830827/')
        print('mean loss:', mean_loss)
        print('mean iou:', mean_iou)

    else:
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer = None
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        best_iou = 0
        # for epoch in range(1, args.epochs + 1):
        #     train(args, model, device, train_loader, optimizer, epoch)
        if args.task == 'water':
            mean_loss, mean_iou = test_water(args, model, device, test_loader, 0)
        else:
            mean_loss, mean_iou = test_roof(args, model, device, test_loader, 0)
        mean_iou = np.array(mean_iou).mean()
            # if mean_iou > best_iou:
            #     print('updating best model...')
            #     best_iou = mean_iou
            #     if args.visualize: visualize(args, model, device, test_loader, 'vis_unet/')
            #     if args.save_model: torch.save(model.state_dict(), 'ckpt.pt')
            # scheduler.step()
        print('final mean iou', mean_iou)


if __name__ == '__main__':
    main()
