#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
# from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Shift Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)

    # evaluation args
    parser.add_argument(
        '--test-dir',
        default='',
        help='path to the test directory')
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):

        arg.model_saved_name = "./save_models/"+arg.Experiment_name
        arg.work_dir = "./work_dir/"+arg.Experiment_name
        self.arg = arg

        self.global_step = 0
        self.load_model()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if not os.path.exists(self.arg.test_dir):
            raise ValueError('Test dir is not exist')

        self.arg.weights = os.path.join(self.arg.test_dir, 'weights.pt')
        # self.global_step = int(arg.weights[:-3].split('-')[-1])
        self.print_log('Load weights from {}.'.format(self.arg.weights))
        if '.pkl' in self.arg.weights:
            with open(self.arg.weights, 'r') as f:
                weights = pickle.load(f)
        else:
            weights = torch.load(self.arg.weights)

        weights = OrderedDict(
            [[k.split('module.')[-1],
                v.cuda(output_device)] for k, v in weights.items()])

        for w in self.arg.ignore_weights:
            if weights.pop(w, None) is not None:
                self.print_log('Sucessfully Remove Weights: {}.'.format(w))
            else:
                self.print_log('Can Not Remove Weights: {}.'.format(w))

        try:
            self.model.load_state_dict(weights)
        except:
            state = self.model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            print('Can not find these weights:')
            for d in diff:
                print('  ' + d)
            state.update(weights)
            self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                data = Variable(
                    data.float().cuda(self.output_device),
                    requires_grad=False)
                label = Variable(
                    label.long().cuda(self.output_device),
                    requires_grad=False)

                with torch.no_grad():
                    output = self.model(data)

                loss = self.loss(output, label)
                score_frag.append(output.data.cpu().numpy())
                loss_value.append(loss.data.cpu().numpy())

                _, predict_label = torch.max(output.data, 1)
                step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            if not 'sample_name' in self.data_loader[ln].dataset.__dict__:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))

            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                score_dict = dict(
                    zip(self.data_loader[ln].dataset.sample_name, score))

            print('Eval Accuracy: ', accuracy)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
            # save results
            with open(os.path.join(self.arg.test_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(score_dict, f)
                self.print_log('saving result at %s' % os.path.join(self.arg.test_dir, 'results.pkl'))


    def start(self):
        self.eval(
            0,
            save_score=self.arg.save_score,
            loader_name=['test'])



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if not os.path.exists(p.test_dir):
        raise ValueError('Test dir is not exist')
    config_file = os.path.join(p.test_dir, 'config.yaml')

    with open(config_file, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
    parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    processor.start()
