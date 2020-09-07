import init_paths
import argparse
import time
import sys
import torch.nn as nn
import torch.optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn
from utils import *
from copy import deepcopy
from validation import validate, validate_pgd
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./writer/LHFAT_CIFAR10/')

cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Adv Training')
    parser.add_argument('--data', default='./data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--output_prefix', default='LHFAT_CIFAR10', type=str,
                        help='prefix used to define output path')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                        help='path to the config file (default: configs.yml)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--t_0', type=float, default=0.9, help=(
            'How far into training to start freezing. Note that this if using' + ' cubic scaling then this is the uncubed value.'))
    parser.add_argument('--scale_lr', type=bool, default=True,
                        help='Scale each layer''s start LR as a function of its t_0 value?')
    parser.add_argument('--no_scale', action='store_false', dest='scale_lr',
                        help='Don''t scale each layer''s start LR as a function of its t_0 value')
    parser.add_argument('--how_scale', type=str, default='cubic', help=(
            'How to relatively scale the schedule of each subsequent layer.' + 'options: linear, squared, cubic.'))
    parser.add_argument('--save_freq', default=1, type=int, help='save frequency')
    parser.add_argument('--eval_adv_freq', default=1, type=int, help='eval adv frequency')
    parser.add_argument('--save_dir', default="trained_models/LHFAT/LHFAT_CIFAR10", help='save path')

    return parser.parse_args()


# Parase config file and initiate logging
configs = parse_config_file(parse_args())
logger = initiate_logger(configs.output_name)
print = logger.info

# total trainable layer block for WRN_34 is 18
n_layer = configs.TRAIN.n_layer
layer_ratio = [0] * n_layer

def main():
    # Scale and initialize the parameters
    best_prec1 = 0
    global model, model_path
    configs.TRAIN.epochs = int(math.ceil(configs.TRAIN.epochs / configs.ADV.n_repeats))
    configs.ADV.fgsm_step /= configs.DATA.max_color_value
    configs.ADV.clip_eps /= configs.DATA.max_color_value

    # Create output folder
    if not os.path.isdir(os.path.join(configs.save_dir, configs.output_name)):
        os.makedirs(os.path.join(configs.save_dir, configs.output_name))

    # Log the config details
    logger.info(pad_str(' ARGUMENTS '))
    for k, v in configs.items(): print('{}: {}'.format(k, v))
    logger.info(pad_str(''))

    print('==> Building model..')
    if configs.TRAIN.arch == 'wideresnet_34':
        from models.wideresnet_FF import WideResNet as WRN
        model = WRN(depth=34, num_classes=10, epochs=configs.TRAIN.epochs, t_0=configs.t_0, widen_factor=10,
            scale_lr=configs.scale_lr, how_scale=configs.how_scale)
        model_path = ''

    # Wrap the model into DataParallel
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    if not configs.resume and os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path,map_location=device_mapping(-1))
        best_acc = checkpoint['acc']
        model.load_state_dict(checkpoint['net'])
        print("=> loaded checkpoint '{}' (ori model trained on epoch {})"
              .format(model_path, checkpoint['epoch']))
        print("=> best_acc '{}'".format(best_acc))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    # Criterion:
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer:
    optimizer = torch.optim.SGD([{'params': m.parameters(), 'lr': m.lr}
                                 for m in model.modules() if hasattr(m,'active')],
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)

    # Resume if a valid checkpoint path is provided
    if configs.resume and not configs.evaluate:
        if os.path.isfile(configs.resume):
            print("=> loading checkpoint '{}'".format(configs.resume))
            checkpoint = torch.load(configs.resume,map_location=device_mapping(-1))
            configs.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(configs.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

    # Initiate data loaders
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # trainset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=configs.DATA.batch_size, shuffle=True,
                                               num_workers=configs.DATA.workers)

    # testnset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=False)

    val_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=configs.DATA.batch_size, shuffle=False,
                                             num_workers=configs.DATA.workers)

    print("trainset_len: %s, testset_len: %s" % (str(trainset.__len__()),
                                                 str(testset.__len__())))

    # If in evaluate mode: perform validation on PGD attacks as well as clean samples
    if configs.evaluate and configs.resume:
        if os.path.isfile(configs.resume):
            print("=> loading checkpoint '{}'".format(configs.resume))
            checkpoint = torch.load(configs.resume,map_location=device_mapping(-1))
            configs.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(configs.resume, checkpoint['epoch']))
            print("=> loaded best_prec1 '{}'".format(best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

        logger.info(pad_str(' Performing PGD Attacks '))
        for pgd_param in configs.ADV.pgd_attack:
            validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs, logger)
        validate(val_loader, model, criterion, configs, logger)
        return

    global layer_ratio
    init_ratio_list = get_init_layer_ratio(model, n_layer)

    import time
    start_time = time.time()

    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        set_layerwise_ratio(model, epoch, init_ratio_list, layer_ratio)

        # ############################ Train #########################
        # train for one epoch
        epoch_start_time = time.time()
        train(train_loader, model, criterion, optimizer, epoch, n_layer)
        logger.info(pad_str("Epoch_%d_time: %.3f" % (epoch, time.time() - epoch_start_time)))
        # ############################ Test #########################
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, configs, logger)

        writer.add_scalar('data/Test_acc', prec1, epoch)
        writer.add_scalar('data/Epoch_time', time.time() - epoch_start_time, epoch)

        # evaluate on ADV
        if epoch%configs.eval_adv_freq==0:
            if epoch > 16:
                pgd_steps = configs.ADV.pgd_attack[0][0]
                step_size = configs.ADV.pgd_attack[0][1]
                PGD_val_prec1 = validate_pgd(val_loader, model, criterion, pgd_steps, step_size, configs, logger)
                writer.add_scalar('data/PGD_acc', PGD_val_prec1, epoch)
                print('Epoch {}, PGD Final Prec@1 {}'.format(epoch,PGD_val_prec1))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch,
            'arch': configs.TRAIN.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, configs.save_freq, os.path.join(configs.save_dir, configs.output_name))

    # Automatically perform PGD Attacks at the end of training
    logger.info(pad_str(' Performing PGD Attacks '))
    for pgd_param in configs.ADV.pgd_attack:
        validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs, logger)

    logger.info(pad_str("Total time: %s" % (time.time() - start_time)))
    writer.close()


# Adversarial Training Module
global_noise_data = torch.zeros([configs.DATA.batch_size, 3, configs.DATA.crop_size, configs.DATA.crop_size]).cuda()


def train(train_loader, model, criterion, optimizer, epoch, n_layer):
    global global_noise_data
    global layer_ratio

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

    lr_coefficient = adjust_lr_coefficient(epoch, configs)

    # reset lr according to lr_ratio
    reset_lr_according_Ratio(model, configs)
    # reset optimizer
    optimizer = torch.optim.SGD([{'params': m.parameters(), 'lr': m.lr * lr_coefficient}
                                 for m in model.modules() if hasattr(m,'active')],
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)

    iters_per_epoch = len(train_loader) - 1

    #　Reset scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iters_per_epoch, eta_min=3e-4*lr_coefficient)

    # init gradout_list and grad_iter_abs at the begining of every epoch
    gradout_list = [0] * n_layer
    grad_iter_abs = [0] * n_layer

    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        scheduler.step()
        global_noise_data = torch.FloatTensor(input.size()).uniform_(-configs.ADV.clip_eps, configs.ADV.clip_eps).cuda()

        for j in range(configs.ADV.n_repeats):
            # Ascend on the global noise
            noise_batch = deepcopy(global_noise_data)
            noise_batch.requires_grad_()
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

    print('Final Train Epoch: [{0}][{1}/{2}]\t'
          'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        epoch, len(train_loader), len(train_loader),
        top1=top1, top5=top5, cls_loss=losses))
    sys.stdout.flush()

    writer.add_scalar('data/Train_loss', losses.avg, epoch)
    writer.add_scalar('data/Train_acc', top1.avg, epoch)

    # calculate layer_ratio at the end of even epoch
    if epoch != configs.TRAIN.epochs - 1 and epoch % 2 == 0:
        layer_ratio = calculate_layer_ratio(epoch, gradout_list, grad_iter_abs)


if __name__ == '__main__':
    main()
