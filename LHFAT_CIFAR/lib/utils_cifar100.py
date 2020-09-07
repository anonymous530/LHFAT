import logging
import os
import datetime
import torchvision.models as models
import math
import torch
import yaml
from easydict import EasyDict
import shutil
import random
import numpy as np


GLOBAL_SEED = 0


def set_seed(seed):
    print('Set seed ', seed, '...')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def init_fn(worker_id):
    np.random.seed(GLOBAL_SEED + worker_id)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = initial_lr * (0.1 ** (epoch // int(math.ceil(30. / n_repeats))))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_lr_coefficient(epoch, configs):
    """Sets the learning rate ratio"""
    if epoch < configs.TRAIN.lr_epoch_schedule[1][0]:
        lr_coefficient = configs.TRAIN.lr_epoch_schedule[0][1]
    elif epoch < configs.TRAIN.lr_epoch_schedule[2][0]:
        lr_coefficient = configs.TRAIN.lr_epoch_schedule[1][1]
    else:
        lr_coefficient = configs.TRAIN.lr_epoch_schedule[2][1]
    print("Learning rate decay. lr_coefficient: %s" % lr_coefficient)

    return lr_coefficient


def reset_lr_according_Ratio(model, configs):
    # reset lr according to lr_ratio
    count_layer = 0
    for i, m in enumerate(model.modules()):
        if hasattr(m, 'active'):
            count_layer += 1
            m.lr = configs.TRAIN.lr * m.lr_ratio
            # print('count_layer: %s, m.lr_ratio: %s, m.lr: %s' % (count_layer, m.lr_ratio, m.lr))


def layerwise_adapt(model, layer_ratio):
    idx = 0
    rank = torch.argsort(torch.tensor(layer_ratio))
    min_k = rank[:5]
    max_k = rank[5:]
    # Loop over all modules
    for m in model.modules():
        if hasattr(m,'active'):
            if idx in min_k:
                m.lr_ratio -= 0.1 * m.lr_ratio
            elif idx in max_k:
                m.lr_ratio += 0.1 * layer_ratio[idx] * m.lr_ratio
            idx += 1
    # print("Layerwise adapted.")


def get_init_layer_ratio(model, n_layer):
    temp_i = 0
    init_ratio_list = [0.0 for _ in range(n_layer)]
    for i, m in enumerate(model.modules()):
        if hasattr(m, 'active'):
            init_ratio_list[temp_i] = m.lr_ratio
            temp_i += 1
    return init_ratio_list


def set_layerwise_ratio(model, epoch, init_ratio_list, layer_ratio):
    if epoch != 0:
        # even epoch，reset ratio
        if epoch % 2 == 0:
            temp_i = 0
            for i, m in enumerate(model.modules()):
                if hasattr(m, 'active'):
                    m.lr_ratio = init_ratio_list[temp_i]
                    temp_i += 1
        else:
            # odd epoch，layer ratio adapt
            layerwise_adapt(model, layer_ratio)


def get_gradients(model, gradout_list, grad_iter_abs):
    i = 0
    for name, m in model.named_modules():
        if hasattr(m,'active'):
            if 'block' in name:
                gradout_list[i] += m.conv2.weight.grad.data
                grad_iter_abs[i] += torch.abs(m.conv2.weight.grad.data)
            else:
                gradout_list[i] += m.weight.grad.data
                grad_iter_abs[i] += torch.abs(m.weight.grad.data)
            i += 1


def calculate_layer_ratio(epoch, gradout_list, grad_iter_abs):
    # abs gradout_list every epoch
    gradout_list = [torch.abs(i) for i in gradout_list]
    # grad_iter_abs denotes abs(grad_iter)

    sum_grad_iter_abs = [float(torch.sum(i)) for i in grad_iter_abs]
    sum_grad_epoch_abs = [float(torch.sum(i)) for i in gradout_list]

    layer_ratio_list = [(i / j) if j != 0 else 0 for i, j in zip(sum_grad_epoch_abs, sum_grad_iter_abs)]
    layer_ratio_list = torch.Tensor(layer_ratio_list)
    layer_ratio_list = (layer_ratio_list - torch.min(layer_ratio_list)) / (
                torch.max(layer_ratio_list) - torch.min(layer_ratio_list))
    layer_ratio_list = layer_ratio_list.tolist()

    return layer_ratio_list


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def initiate_logger(output_path):
    if not os.path.isdir(os.path.join('output', output_path)):
        os.makedirs(os.path.join('output', output_path))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join('output', output_path, 'log.txt'), 'w'))
    logger.info(pad_str(' LOGISTICS '))
    logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logger.info('Output Name: {}'.format(output_path))
    logger.info('User: {}'.format(os.getenv('USER')))
    return logger


def get_model_names():
    return sorted(name for name in models.__dict__
                  if name.islower() and not name.startswith("__")
                  and callable(models.__dict__[name]))


def pad_str(msg, total_len=70):
    rem_len = total_len - len(msg)
    return '*' * int(rem_len / 2) + msg + '*' * int(rem_len / 2) \



def parse_config_file(args):
    with open(args.config) as f:
        config = EasyDict(yaml.load(f))

    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v

    # Add the output path
    config.output_name = '{:s}_step{:d}_eps{:d}_repeat{:d}'.format(args.output_prefix,
                                                                   int(config.ADV.fgsm_step), int(config.ADV.clip_eps),
                                                                   config.ADV.n_repeats)
    return config


def save_checkpoint(state, is_best, epoch, save_freq, filepath):
    filename = os.path.join(filepath, 'checkpoint.pth.tar')
    # Save model
    torch.save(state, filename)
    if epoch % save_freq ==0:
        filename = os.path.join(filepath, 'checkpoint_'+str(epoch)+'.pth.tar')
        torch.save(state, filename)
    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))


def device_mapping(cuda_device: int):
    """
    In order to `torch.load()` a GPU-trained model onto a CPU (or specific GPU),
    you have to supply a `map_location` function. Call this with
    the desired `cuda_device` to get the function that `torch.load()` needs.
    """

    def inner_device_mapping(storage: torch.Storage, location) -> torch.Storage:
        if cuda_device >= 0:
            return storage.cuda(cuda_device)
        else:
            return storage

    return inner_device_mapping



scale_fn = {'linear': lambda x: x, 'squared': lambda x: x ** 2, 'cubic': lambda x: x ** 3}
