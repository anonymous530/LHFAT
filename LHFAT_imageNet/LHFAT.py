# This module is adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import init_paths
import argparse
import os
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import numpy as np
from utils import *
from validation import validate, validate_pgd
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./writer/LHFAT_imagenet/')

os.environ['TORCH_HOME'] = 'models/pretrained_models'
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

set_seed(GLOBAL_SEED)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--output_prefix', default='LHFAT_imagenet', type=str,
                        help='prefix used to define output path')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                        help='path to the config file (default: configs.yml)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--save_freq', default=1, type=int, help='save frequency')
    parser.add_argument('--t_0', type=float, default=0.9, help=(
            'How far into training to start freezing. Note that this if using' + ' cubic scaling then this is the uncubed value.'))
    parser.add_argument('--scale_lr', type=bool, default=True,
                        help='Scale each layer''s start LR as a function of its t_0 value?')
    parser.add_argument('--no_scale', action='store_false', dest='scale_lr',
                        help='Don''t scale each layer''s start LR as a function of its t_0 value')
    parser.add_argument('--how_scale', type=str, default='cubic', help=(
            'How to relatively scale the schedule of each subsequent layer.' + 'options: linear, squared, cubic.'))
    parser.add_argument('--const_time', type=bool, default=False,
                        help='Scale the #epochs as a function of ice to match wall clock time.')
    parser.add_argument('--eval_add_freq', type=int, default=1, help='eval adv frequency')
    return parser.parse_args()


# Parase config file and initiate logging
configs = parse_config_file(parse_args())
logger = initiate_logger(configs.output_name)
print = logger.info

# n_layer = 18   # for ResNet50
n_layer = 35     # for ResNet101
layer_ratio = [0] * n_layer


def main():
    # Scale and initialize the parameters
    best_prec1 = 0
    configs.TRAIN.epochs = int(math.ceil(configs.TRAIN.epochs / configs.ADV.n_repeats))
    configs.ADV.fgsm_step /= configs.DATA.max_color_value
    configs.ADV.clip_eps /= configs.DATA.max_color_value

    # Create output folder
    if not os.path.isdir(os.path.join('trained_models/LHFAT', configs.output_name)):
        os.makedirs(os.path.join('trained_models/LHFAT', configs.output_name))

    # Log the config details
    logger.info(pad_str(' ARGUMENTS '))
    for k, v in configs.items(): print('{}: {}'.format(k, v))
    logger.info(pad_str(''))

    # Create the model
    if configs.pretrained:
        print("=> using pre-trained model '{}'".format(configs.TRAIN.arch))
        # model = models.__dict__[configs.TRAIN.arch](pretrained=True)
        if configs.TRAIN.arch == 'resnet50':
            from models.resnet_ratio import resnet50 as resnet50_ratio
            model = resnet50_ratio(t_0=configs.t_0, scale_lr=configs.scale_lr, how_scale=configs.how_scale, const_time=configs.const_time)
            model.load_state_dict(torch.load('./models/pretrained_models/resnet50-19c8e357.pth'))

        elif configs.TRAIN.arch == 'resnet101':
            from models.resnet_ratio import resnet101 as resnet101_ratio
            model = resnet101_ratio(t_0=configs.t_0, scale_lr=configs.scale_lr, how_scale=configs.how_scale, const_time=configs.const_time)
            model.load_state_dict(torch.load('./models/pretrained_models/resnet101-5d3b4d8f.pth'))

        elif configs.TRAIN.arch == 'resnet152':
            from models.resnet_ratio import resnet152 as resnet152_ratio
            model = resnet152_ratio(t_0=configs.t_0, scale_lr=configs.scale_lr, how_scale=configs.how_scale, const_time=configs.const_time)
            model.load_state_dict(torch.load('./models/pretrained_models/resnet152-b121ed2d.pth'))

    else:
        print("=> creating model '{}'".format(configs.TRAIN.arch))
        # model = models.__dict__[configs.TRAIN.arch]()
        if configs.TRAIN.arch == 'resnet50':
            from models.resnet_ratio import resnet50 as resnet50_ratio
            model = resnet50_ratio(t_0=configs.t_0, scale_lr=configs.scale_lr, how_scale=configs.how_scale,
                               const_time=configs.const_time)
        elif configs.TRAIN.arch == 'resnet101':
            from models.resnet_ratio import resnet101 as resnet101_ratio
            model = resnet101_ratio(t_0=configs.t_0, scale_lr=configs.scale_lr, how_scale=configs.how_scale,
                                    const_time=configs.const_time)
        elif configs.TRAIN.arch == 'resnet152':
            from models.resnet_ratio import resnet152 as resnet152_ratio
            model = resnet152_ratio(t_0=configs.t_0, scale_lr=configs.scale_lr, how_scale=configs.how_scale,
                                    const_time=configs.const_time)
    # Wrap the model into DataParallel
    model = torch.nn.DataParallel(model).cuda()

    # Criterion:
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer:
    optimizer = torch.optim.SGD([{'params': m.parameters(), 'lr': m.lr}
                                 for m in model.module.modules() if hasattr(m, 'active')],
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)

    # Resume if a valid checkpoint path is provided
    if configs.resume:
        if os.path.isfile(configs.resume):
            print("=> loading checkpoint '{}'".format(configs.resume))
            checkpoint = torch.load(configs.resume, map_location=device_mapping(-1))
            configs.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(configs.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

    # Initiate data loaders
    traindir = os.path.join(configs.data, 'train')
    valdir = os.path.join(configs.data, 'val')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(configs.DATA.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs.DATA.batch_size, shuffle=True,
        num_workers=configs.DATA.workers, pin_memory=True, sampler=None, worker_init_fn=init_fn)

    normalize = transforms.Normalize(mean=configs.TRAIN.mean,
                                     std=configs.TRAIN.std)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(configs.DATA.img_size),
            transforms.CenterCrop(configs.DATA.crop_size),
            transforms.ToTensor(),
        ])),
        batch_size=configs.DATA.batch_size, shuffle=False,
        num_workers=configs.DATA.workers, pin_memory=True, worker_init_fn=init_fn)

    # If in evaluate mode: perform validation on PGD attacks as well as clean samples
    if configs.evaluate:
        logger.info(pad_str(' Performing PGD Attacks '))
        for pgd_param in configs.ADV.pgd_attack:
            validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs, logger)
        validate(val_loader, model, criterion, configs, logger)
        return

    global layer_ratio
    # recording initial m.lr_raio_list
    init_ratio_list = get_init_layer_ratio(model, n_layer)

    import time
    start_time = time.time()

    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        set_layerwise_ratio(model, epoch, init_ratio_list, layer_ratio)

        # ############################ Train #########################
        epoch_start_time = time.time()
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, n_layer)
        # evaluate on validation set
        logger.info("Epoch_%d_time: %.3f" % (epoch, time.time() - epoch_start_time))
        prec1 = validate(val_loader, model, criterion, configs, logger)
        writer.add_scalar('data/Test_acc', prec1, epoch)
        writer.add_scalar('data/Epoch_time', time.time() - epoch_start_time, epoch)

        # evaluate on ADV
        if epoch%configs.eval_adv_freq==0:
            PGD_val_prec1 = validate_pgd(val_loader, model, criterion, 10, 0.00392156862, configs, logger)
            writer.add_scalar('data/PGD_acc', PGD_val_prec1, epoch)
            print('Epoch {}, PGD Final Prec@1 {}'.format(epoch,PGD_val_prec1))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint_with_freq({
            'epoch': epoch,
            'arch': configs.TRAIN.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, configs.save_freq, os.path.join('trained_models/LHFAT', configs.output_name))

    logger.info(pad_str("Total train time: %s" % (time.time() - start_time)))

    # Automatically perform PGD Attacks at the end of training
    logger.info(pad_str(' Performing PGD Attacks '))
    for pgd_param in configs.ADV.pgd_attack:
        validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs, logger)

    logger.info(pad_str("Total time: %s" % (time.time() - start_time)))

    writer.close()

# Adversarial Training Module
global global_noise_data
global_noise_data = torch.zeros([configs.DATA.batch_size, 3, configs.DATA.crop_size, configs.DATA.crop_size]).cuda()

def train(train_loader, model, criterion, optimizer, epoch, n_layer):
    global layer_ratio
    global global_noise_data
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    lr_coefficient = adjust_lr_coefficient(epoch)
    # reset lr according to lr_ratio
    reset_lr_according_Ratio(model, configs)
    # reset optimizer
    optimizer = torch.optim.SGD([{'params': m.parameters(), 'lr': m.lr * lr_coefficient}
                                 for m in model.module.modules() if hasattr(m, 'active')],
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)

    iters_per_epoch = len(train_loader) - 1
    # 　Reset scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iters_per_epoch, eta_min=5e-5)

    # init gradout_list and grad_iter_abs at the begining of every epoch
    gradout_list = [0] * n_layer
    grad_iter_abs = [0] * n_layer

    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        for j in range(configs.ADV.n_repeats):
            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)
            output = model(in1)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            if epoch != configs.TRAIN.epochs - 1 and epoch % 2 == 0:
                get_gradients(model, gradout_list, grad_iter_abs)

            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-configs.ADV.clip_eps, configs.ADV.clip_eps)

            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs.TRAIN.print_freq == 0:
                print('Train Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, top1=top1, top5=top5, cls_loss=losses))
                sys.stdout.flush()

        scheduler.step()

    print('Train Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        epoch, len(train_loader), len(train_loader), batch_time=batch_time,
        data_time=data_time, top1=top1, top5=top5, cls_loss=losses))
    sys.stdout.flush()

    writer.add_scalar('data/Train_loss', losses.avg, epoch)
    writer.add_scalar('data/Train_acc', top1.avg, epoch)

    # 偶数epoch结尾开始计算，奇数epoch起始adapt
    if epoch != configs.TRAIN.epochs - 1 and epoch % 2 == 0:
        layer_ratio = calculate_layer_ratio(gradout_list, grad_iter_abs)


if __name__ == '__main__':
    main()
