import argparse
import os
import time
import shutil
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.backends.cudnn as cudnn
import random
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from logger import Logger
from models import *
from data_loader import data_loader
from helper import AverageMeter, save_checkpoint,save_completecheckpoint, accuracy, adjust_learning_rate, adjust_box

#from utils import progress_bar

import yaml
import json
import io_utils
from torch.utils.tensorboard import SummaryWriter

import hessian_spectrum
import timm
import swanlab


model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'vit_base'
]

loss_min = 0.3
loss_max = 6.8

print("init amp")
scaler = torch.cuda.amp.GradScaler()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/yszhang/datasets/imagenet/', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batchsize', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--dataset_mode', default='full', type=str, 
                    help="use full dataset, uniform subset or 1/k distributed subset")
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--warmup_epochs', default=0, type=int, 
                    help='warm up epochs for lr scheduler')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1 of adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 of adam')
parser.add_argument('--momentum', default=0.95, type=float, metavar='M',
                    help='momentum for sgd')
parser.add_argument('--wd', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_false',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--opt', default='adam', type=str, help=' optimizer, adam or sgd or others')
parser.add_argument('--seed', default=32, type=int,help='seed for training. ')
parser.add_argument('--epsilon', '-eps', default=1e-8, type=float, help='epsilon for stability')
parser.add_argument('--comment', '-comment', default='-', type=str, help='some additional comments')
parser.add_argument('--use_minibatch', action='store_true', help='Set the flag to True')

parser.add_argument('--load_iter', type = int, default=0, help='load ')

parser.add_argument('--ckpt_std', default='epoch', type=str, help='save ckpt by epoch or loss')

parser.add_argument('--gradient_accumulation_steps', type = int, default = 0 )
parser.add_argument('--shuffle',action='store_true', help = 'whether use shuffle in training data.')
parser.add_argument('--plot_hessian',action='store_true', help = 'whether plot hessian or not')


def main():
    global args
    args = parser.parse_args()

    print('w',os.environ.get('WORLD_SIZE'))
    print('w2', int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1)
    print('ngpus_per_node = torch.cuda.device_count()',torch.cuda.device_count())

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    last_time = time.time()
    if args.arch == 'alexnet':
        model = alexnet(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_0':
        model = squeezenet1_0(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_1':
        model = squeezenet1_1(pretrained=args.pretrained)
    elif args.arch == 'densenet121':
        model = densenet121(pretrained=args.pretrained)
    elif args.arch == 'densenet169':
        model = densenet169(pretrained=args.pretrained)
    elif args.arch == 'densenet201':
        model = densenet201(pretrained=args.pretrained)
    elif args.arch == 'densenet161':
        model = densenet161(pretrained=args.pretrained)
    elif args.arch == 'vgg11':
        model = vgg11(pretrained=args.pretrained)
    elif args.arch == 'vgg13':
        model = vgg13(pretrained=args.pretrained)
    elif args.arch == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
    elif args.arch == 'vgg19':
        model = vgg19(pretrained=args.pretrained)
    elif args.arch == 'vgg11_bn':
        model = vgg11_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg13_bn':
        model = vgg13_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg19_bn':
        model = vgg19_bn(pretrained=args.pretrained)
    elif args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
    elif args.arch == 'vit_base': # not working
        # model = timm.create_model('vit_base_patch16_224',pretrained=False)
        model = vit_b_16(pretrained=args.pretrained)
    else:
        raise NotImplementedError

    model.cuda()

    'parallel'
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    #model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon,
                               weight_decay=args.wd, amsgrad=False)

    elif args.opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon,
                                weight_decay=args.wd, amsgrad=False)
                                     
    
    # 创建学习率调度器 - CosineAnnealing with Warmup
    warmup_epochs = args.warmup_epochs
    scheduler1 = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], 
                            milestones=[warmup_epochs])
    
    # optionlly resume from a checkpoint
    if args.load_iter == 0:
        args.resume = ''

    if args.resume:
        if args.ckpt_std == "epoch":
            file_name = args.resume + args.arch + args.opt + args.dataset_mode + '_ckpt_' + str(args.load_iter)+'.pth'
        if args.ckpt_std == "loss":
            file_name = args.resume
        print("try to load ckpt from", file_name)
        if os.path.isfile(file_name):
            print("=> loading checkpoint '{}'".format(file_name))
            checkpoint = torch.load(file_name)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(file_name, checkpoint['epoch']))
        
            print('resumed from: ', file_name)
            time.sleep(3)

        else:
            print("=> no checkpoint found")
            assert 0

    os.makedirs('log', exist_ok= True)
    if args.opt == 'sgd':
        save_name = 'ImageNET-model-{}-opt-{}-lr-{}-momentum-{}-wd-{}-bs-{}-seed-{}-comment-{}-dsmode-{}'.format(args.arch, 
                                                                                                                args.opt, 
                                                                                                                args.lr, 
                                                                                                                args.momentum, 
                                                                                                                args.wd, 
                                                                                                                args.batchsize, 
                                                                                                                args.seed, 
                                                                                                                args.comment, 
                                                                                                                args.dataset_mode)
    else:
        save_name = 'ImageNET-model-{}-opt-{}-lr-{}-beta1-{}-beta2-{}-wd-{}-bs-{}-seed-{}-comment-{}-dsmode-{}'.format(args.arch, 
                                                                                                        args.opt, 
                                                                                                        args.lr, 
                                                                                                        args.beta1, 
                                                                                                        args.beta2, 
                                                                                                        args.wd, 
                                                                                                        args.batchsize, 
                                                                                                        args.seed, 
                                                                                                        args.comment, 
                                                                                                        args.dataset_mode)
    'set up logger: change name here'
    save_dir = os.path.join('log', save_name)

    os.makedirs(save_dir, exist_ok=True)
    logger = Logger('{}/logger.txt'.format(save_dir), title='logger')
    logger.set_names(['epoch', 'trainloss', 'testloss','trainacc','testacc', 'current_lr'])
    writer = SummaryWriter(save_dir)
    io_utils.save_code(save_dir)


    # cudnn.benchmark = True

    # Data loading
    print('start loading data')
    ngpus_per_node = torch.cuda.device_count()
    #args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    #print('load data worker',args.workers)
    print('load data pin_mem', args.pin_memory)


    if args.plot_hessian:
        args.shuffle = False
    else: args.shuffle = True

    if args.plot_hessian:
        train_loader, val_loader = data_loader(
            args.data, args.batchsize, args.workers, 
            args.pin_memory, shuffle=args.shuffle, 
            mode=args.dataset_mode, 
            save_path=save_dir
        )
    else:
        train_loader, val_loader = data_loader(
            args.data, args.batchsize, args.workers, 
            args.pin_memory, shuffle=args.shuffle, 
            mode=args.dataset_mode, save_path=save_dir
        )
    print('loading complete')


    if args.evaluate:
        test(val_loader, model)
        return
    
    print("args.plot_hessian:", args.plot_hessian)
    if args.plot_hessian:
        plot_hessian(args, model, train_loader)

    else:
        training(args, optimizer, scheduler, save_dir, train_loader, val_loader, model, criterion, writer,logger)
        
        
def log_weight_grad_norms(model):
    """
    统计 weight 的梯度范数
    会把每一层的 grad_norm 和一个 total grad_norm 打到 SwanLab。
    """
    grad_logs = {}
    total_norm_sq = 0.0

    for name, p in model.named_parameters():
        # 没梯度/不需要梯度的跳过
        if (p.grad is None) or (not p.requires_grad):
            continue
        # 跳过 bias
        if 'bias' in name:
            continue

        # L2 范数
        g = p.grad.detach()
        n = g.norm(2).item()
        grad_logs[f"grad_norm/{name}"] = n

        total_norm_sq += n ** 2

    # 一个全局的梯度范数
    if total_norm_sq > 0:
        grad_logs["grad_norm/total"] = total_norm_sq ** 0.5

    if grad_logs:  # 有东西再打 log
        swanlab.log(grad_logs)



def plot_hessian(args, model, train_loader):
    print("Start Plot Hessian ...")
    if args.opt == 'sgd':
        comment = 'model-{}-opt-{}-lr-{}-momentum-{}-wd-{}-seed-{}-dsmode-{}'.format(
            args.arch, args.opt, args.lr, 
            args.momentum, args.wd, args.seed, 
            args.dataset_mode
        )
    else:
        comment = 'model-{}-opt-{}-lr-{}-beta1-{}-beta2-{}-wd-{}-seed-{}-dsmode-{}'.format(
            args.arch, args.opt, args.lr, 
            args.beta1, args.beta2, args.wd, 
            args.seed, args.dataset_mode
        )

    sigma = 1e-5
    batch_size = args.batchsize
    gradient_accumulation_steps = args.gradient_accumulation_steps # 256 *30
    load_iter = args.load_iter

    sample_layer = []
    #vit 
    if args.arch == 'vit_base':
        sample_layer = [
            "module.conv_proj.weight",
            "module.encoder.layers.encoder_layer_0.self_attention.c_attn.weight",
            "module.encoder.layers.encoder_layer_0.self_attention.c_proj.weight",
            "module.encoder.layers.encoder_layer_0.mlp.0.weight",
            "module.encoder.layers.encoder_layer_0.mlp.3.weight",
            "module.encoder.layers.encoder_layer_6.self_attention.c_attn.weight",
            "module.encoder.layers.encoder_layer_6.self_attention.c_proj.weight",
            "module.encoder.layers.encoder_layer_6.mlp.0.weight",
            "module.encoder.layers.encoder_layer_6.mlp.3.weight",
            "module.encoder.layers.encoder_layer_11.self_attention.c_attn.weight",
            "module.encoder.layers.encoder_layer_11.self_attention.c_proj.weight",
            "module.encoder.layers.encoder_layer_11.mlp.0.weight",
            "module.encoder.layers.encoder_layer_11.mlp.3.weight",
            "module.heads.head.weight"
        ]
        last_layer = "module.heads.head.weight"
    elif 'resnet' in args.arch or 'vgg' in args.arch:
        for name, _ in model.named_parameters():
            if ('bn' not in name) and ('bias' not in name) and ('downsample' not in name):
                sample_layer.append(name)
        last_layer = "module.fc.weight"

    print(f"sample layers are {sample_layer}")
    
    hessian = hessian_spectrum.Hessian(model=model, 
                                       m=100, 
                                       num_v=1, 
                                       ckpt_iteration=load_iter, 
                                       use_minibatch=args.use_minibatch, 
                                       gradient_accumulation_steps=gradient_accumulation_steps, 
                                       train_data=train_loader, 
                                       batch_size=batch_size, 
                                       sample_layer=sample_layer, 
                                       sigma=sigma, 
                                       comment=comment)
    

    hessian.get_spectrum(layer_by_layer=True)
    hessian.load_curve(layer_by_layer=True)
    avg_dist = plot_hetero_heatmap(hessian, metric="kl", mode="arch")
    print("kl, arch:", avg_dist)
    avg_dist = plot_hetero_heatmap(hessian, metric="js", mode="arch")
    print("js, arch:", avg_dist)

    
#     hessian.calc_classwise_hessian_spectrum(layer_name=last_layer)
#     hessian.plot_classwise_esd()
#     plot_classwise_norms(args, hessian)
#     avg_dist = plot_hetero_heatmap(hessian, metric="js", mode="last_layer")
#     print("js, last_layer:", avg_dist)

#     hessian.calc_last_layer_subset_hessian_heatmap(layer_name=last_layer, 
#                                                    class_indices=np.array([0]))



def plot_hetero_heatmap(hessian, metric="kl", mode="arch"):
    assert metric in ("kl", "js"), "metric must be kl or js"
    assert mode in ("arch", "headwise_attn", "last_layer"), "mode must be arch or last_layer"
    
    if mode == "arch":
        weight_dir = os.path.join(hessian.file_dir, "weights_layer.json")
        value_dir  = os.path.join(hessian.file_dir, "values_layer.json")
        missing_hint = "请先运行 get_spectrum_layer_by_layer()"
    else:  # last_layer
        weight_dir = os.path.join(hessian.file_dir, "classwise_weights.json")
        value_dir  = os.path.join(hessian.file_dir, "classwise_values.json")
        missing_hint = "请先运行 calc_classwise_hessian_spectrum() 生成 classwise_* 文件"

    if not os.path.exists(weight_dir):
        raise FileNotFoundError(f"{weight_dir} 不存在，{missing_hint}")
    with open(weight_dir, 'r') as f:
        weights_dic = json.load(f)

    if not os.path.exists(value_dir):
        raise FileNotFoundError(f"{value_dir} 不存在，{missing_hint}")
    with open(value_dir, 'r') as f:
        values_dic = json.load(f)

    layer_name_list = list(weights_dic.keys())
    n = len(layer_name_list)

    # ====== 分支一：last_layer & KL → 直方图离散版 SKL（只改这里） ======
    if mode == "last_layer" and metric == "kl":
        assert 0, "not supported"
    
    # ========= 新分支：last_layer & JS → 对数域能量直方图 + JS distance =========
    if mode == "last_layer" and metric == "js":
        NUM_BINS   = 128
        EPS        = 1e-12         # 处理零/负特征值
        SMOOTHING  = 1e-12         # Dirichlet 平滑，防止零概率
        BIN_METHOD = "linear"    # "quantile" 或 "linear"
        JS_MAX     = float(np.sqrt(np.log(2.0)))  # JS distance ∈ [0, sqrt(ln 2)]

        # 1) 在 log(λ+eps) 域构建公共分箱边界
        log_edges = build_common_log_bins(values_dic, num_bins=NUM_BINS, eps=EPS, method=BIN_METHOD)

        # 2) 逐类构建“能量直方图”并 L1 归一化为概率
        #    注意：能量质量 = weights * eigenvalues（如果没有权重就用 eigenvalues 本身）
        P_rows = []
        for k in layer_name_list:
            vals = np.asarray(values_dic[k], dtype=float).ravel()
            vals = np.clip(vals, 0.0, None)                         # 负值当 0，稳健起见
            logs = np.log(vals + EPS)                               # log 域定位分箱

            if k in weights_dic and weights_dic[k] is not None:
                w = np.asarray(weights_dic[k], dtype=float).ravel()
                if w.shape != vals.shape:
                    raise ValueError(f"{k}: weights 与 values 形状不一致：{w.shape} vs {vals.shape}")
                mass = w * vals                                     # 能量质量 = 权重 × λ
            else:
                mass = vals                                         # 无权重则用 λ 自身

            # 能量直方图：在 log 域分箱，但质量累加的是“原始 λ 的能量 mass”
            hist, _ = np.histogram(logs, bins=log_edges, weights=mass, density=False)
            hist = hist.astype(np.float64) + SMOOTHING              # Dirichlet 平滑
            total = hist.sum()

            if total <= 0:
                # 退化：全 0 的情形用均匀分布兜底
                p = np.full_like(hist, 1.0 / len(hist), dtype=np.float64)
            else:
                p = (hist / total).astype(np.float64)

            # 再次数值裁剪并归一化，确保严格是概率
            p = np.clip(p, EPS, None)
            p = (p / p.sum()).astype(np.float64)
            P_rows.append(p)

        P = np.vstack(P_rows)   # (C, B)
        n = len(layer_name_list)

        # 3) JS distance 矩阵：JS(P,Q) = H(0.5(P+Q)) - 0.5(H(P)+H(Q))；distance=√JS
        def row_entropy(Pmat, eps=EPS):
            Psafe = np.clip(Pmat, eps, None)
            Psafe = Psafe / Psafe.sum(axis=1, keepdims=True)
            return -(Psafe * np.log(Psafe)).sum(axis=1)

        H_row = row_entropy(P)  # 每行熵
        dist_table = np.zeros((n, n), dtype=np.float64)

        print('calculating JS distance (energy-hist on log-spectrum)')
        for i in range(n):
            Pi = P[i]
            for j in range(i, n):
                Pj = P[j]
                M  = 0.5 * (Pi + Pj)
                M  = np.clip(M, EPS, None)
                M  = M / M.sum()
                Hm = -(M * np.log(M)).sum()
                js_div = Hm - 0.5 * (H_row[i] + H_row[j])          # ∈ [0, ln 2]
                js_dist = float(np.sqrt(max(js_div, 0.0)))          # JS distance
                dist_table[i, j] = dist_table[j, i] = js_dist

        # 4) 保存与作图
        avg_dist = dist_table[np.tril_indices(n, k=-1)].mean() if n > 1 else 0.0
        dist_matrix_path = os.path.join(hessian.file_dir, f'{mode}_{metric}_dist_matrix.npy')
        np.save(dist_matrix_path, dist_table)
        print(f"Distance matrix saved to {dist_matrix_path}")

        mask = np.triu(np.ones_like(dist_table, dtype=bool), k=1)
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            dist_table, cmap="coolwarm", mask=mask, square=True,
            cbar_kws={"label": "Jensen–Shannon Distance"},
            vmin=0.0, vmax=JS_MAX, xticklabels=False, yticklabels=False
        )
        #vmax=JS_MAX, 
        plt.title("Heterogeneity Heatmap (JS Distance on Log-Spectrum Energy)")
        save_path = os.path.join(hessian.file_dir, f'hetero_heatmap_{mode}_{metric}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()

        with open(os.path.join(hessian.file_dir, f"hetero_mean_{mode}_{metric}.txt"), "w") as f:
            f.write(f"Mean JS distance of {mode} (lower triangle, non-diagonal): {avg_dist:.6f}\n")

        print(f"Heatmap saved to {save_path}")
        return avg_dist
        

    # ========== 其它情况 → 走你原来的插值/KDE 逻辑 ==========
    dist_table = np.zeros((n, n))
    pre_count = n * (n + 1) // 2
    total_dist = 0.0
    count = 0
    print('calculating distance')
    for idx, layer_name1 in enumerate(layer_name_list):
        for jdx in range(idx, n):  # 只计算上三角（包括对角线）
            layer_name2 = layer_name_list[jdx]
            print(f"{count} / {pre_count}")
            weights1 = np.array(weights_dic[layer_name1])
            values1  = np.array(values_dic[layer_name1])
            weights2 = np.array(weights_dic[layer_name2])
            values2  = np.array(values_dic[layer_name2])
            distance = interpolate_couple(
                weights1, values1, weights2, values2,
                window_extend=1, hessian=hessian, metric=metric
            )
            dist_table[idx, jdx] = distance
            dist_table[jdx, idx] = distance  # 对称
            total_dist += distance
            count += 1

    avg_dist = total_dist / max(1, count)
    print("avg", avg_dist)

    dist_matrix_path = os.path.join(hessian.file_dir, f'{mode}_{metric}_dist_matrix.npy')
    np.save(dist_matrix_path, dist_table)
    print(f"Distance matrix saved to {dist_matrix_path}")

    names = [name.removeprefix('module.') for name in layer_name_list]
    mask = np.triu(np.ones_like(dist_table, dtype=bool), k=1)
    plt.figure(figsize=(8, 6))
    cbar_label = "Symmetric KL Distance" if metric == "kl" else "Jensen-Shannon Distance"
    title_txt  = f"Heterogeneity Heatmap ({cbar_label})"

    heatmap_kwargs = dict(
        cmap="coolwarm",
        mask=mask,
        square=True,
        cbar_kws={"label": cbar_label},
    )
    if metric == "js":
        heatmap_kwargs.update(vmin=0.0, vmax=float(np.sqrt(np.log(2.0))))

    if mode == "arch":
        ax = sns.heatmap(dist_table, xticklabels=names, yticklabels=names, **heatmap_kwargs)
    else:
        ax = sns.heatmap(dist_table, xticklabels=False, yticklabels=False, **heatmap_kwargs)

    plt.title(title_txt)
    save_path = os.path.join(hessian.file_dir, f'hetero_heatmap_{mode}_{metric}.png')
    plt.savefig(save_path, dpi=200)
    plt.close()

    tril_indices = np.tril_indices(n, k=-1)
    hetero_mean = np.mean(dist_table[tril_indices]) if len(tril_indices[0]) > 0 else 0.0
    metric_name = "Symmetric KL" if metric == "kl" else "JS"
    with open(os.path.join(hessian.file_dir, f"hetero_mean_{mode}_{metric}.txt"), "w") as f:
        f.write(f"Mean {metric_name} distance of {mode} (lower triangle, non-diagonal): {hetero_mean:.4f}\n")

    print(f"Heatmap saved to {save_path}")
    return hetero_mean


def build_common_log_bins(values_dic, num_bins=128, eps=1e-12, method="quantile"):
    """
    在 log(λ+eps) 域为所有类别的特征值构造统一的分箱边界。
    - method="quantile": 用分位数分箱，能缓解分布不均导致的空桶
    - method="linear"  : 等宽分箱（在 log 域上）
    返回：长度 = num_bins+1 的严格递增边界数组
    """
    all_vals = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values_dic.values()])
    all_vals = np.clip(all_vals, 0.0, None)
    z = np.log(all_vals + eps)

    z_min, z_max = float(np.min(z)), float(np.max(z))
    if z_min == z_max:
        z_min -= 1e-12
        z_max += 1e-12

    if method == "quantile":
        edges = np.quantile(z, np.linspace(0.0, 1.0, num_bins + 1))
        # 若重复边界过多，回退到线性等宽
        if np.unique(edges).size < num_bins + 1:
            edges = np.linspace(z_min, z_max, num_bins + 1)
    else:
        edges = np.linspace(z_min, z_max, num_bins + 1)

    # 保证严格递增（添加极小递增微扰）
    edges = np.maximum.accumulate(edges + np.linspace(0, 1e-15, edges.size))
    return edges

    
    
def interpolate_couple(weights1, values1, weights2, values2, window_extend, hessian, metric):

    num_v = hessian.num_v
    
    n_grid = 500000
    left_boundary = min(np.mean(np.min(values1,axis=1)), np.mean(np.min(values2, axis = 1))) - window_extend
    right_boundary= max(np.mean(np.max(values1,axis=1)), np.mean(np.max(values2, axis = 1))) + window_extend
    grid = np.linspace(left_boundary, right_boundary, n_grid).tolist()
    density_all1 = np.zeros((num_v, n_grid))
    density_all2 = np.zeros((num_v, n_grid))
    for k in range(num_v):
        for idx, t in enumerate(grid):
            values_each_v_t1 = hessian.gaussian_density(t, values1[k, :])
            density_each_v_t1 = np.sum(values_each_v_t1 * weights1[k])
            density_all1[k,idx] = density_each_v_t1
            values_each_v_t2 = hessian.gaussian_density(t, values2[k])
            density_each_v_t2 = np.sum(values_each_v_t2 * weights2[k])
            density_all2[k,idx] = density_each_v_t2

    density_avg1 = np.nanmean(density_all1, axis = 0)
    density_avg2 = np.nanmean(density_all2, axis = 0)
    
    if metric == "kl":
        density_avg1[density_avg1 == 0] = 1e-258
        density_avg2[density_avg2 == 0] = 1e-258

        log_density1 = np.log(density_avg1)
        log_density2 = np.log(density_avg2)

        log_norm_fact1 = np.log(np.sum(density_avg1)) + np.log(grid[1]-grid[0])
        log_norm_fact2 = np.log(np.sum(density_avg2)) + np.log(grid[1]-grid[0])
        log_density_avg1_norm = log_density1 - log_norm_fact1
        log_density_avg2_norm = log_density2 - log_norm_fact2

        kl_divergence1 = np.sum(np.exp(log_density_avg1_norm) * (log_density_avg1_norm - log_density_avg2_norm)) * (grid[1]-grid[0])
        kl_divergence2 = np.sum(np.exp(log_density_avg2_norm) * (log_density_avg2_norm - log_density_avg1_norm)) * (grid[1]-grid[0])

        distance = (kl_divergence1 + kl_divergence2)/2
        
    elif metric == "js":
        dx = grid[1]-grid[0]
        div = js_divergence_from_densities(density_avg1, density_avg2, dx) # Xinlu: 这里取的不是JS距离，是JS散度
        distance = np.sqrt(div)

    else:
        assert 0
    
    return distance


def js_divergence_from_densities(p, q, dx, eps=1e-12):
    """p, q: 1D nonnegative arrays on the same grid (unnormalized ok)."""
    p = p + eps
    q = q + eps
    # normalize to pdfs
    p = p / (p.sum() * dx)
    q = q / (q.sum() * dx)
    m = 0.5 * (p + q)
    # JS = 0.5 * (KL(p||m) + KL(q||m))
    kl_pm = np.sum(p * (np.log(p) - np.log(m))) * dx
    kl_qm = np.sum(q * (np.log(q) - np.log(m))) * dx
    return 0.5 * (kl_pm + kl_qm)



def _linregress_np(x, y):
    """
    纯 numpy 线性回归：y = a + b x
    返回：b, a, R2, (b_low, b_high)
    95% CI 用正态近似（1.96），n 较小时可换 t 分布（如装了 SciPy）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    xbar = x.mean()
    ybar = y.mean()
    Sxx = np.sum((x - xbar)**2)
    Sxy = np.sum((x - xbar)*(y - ybar))
    b = Sxy / (Sxx + 1e-20)
    a = ybar - b * xbar
    yhat = a + b * x
    resid = y - yhat
    rss = np.sum(resid**2)
    tss = np.sum((y - ybar)**2) + 1e-20
    R2 = 1.0 - rss / tss
    dof = max(n - 2, 1)
    s_err = np.sqrt(rss / dof)
    se_b = s_err / np.sqrt(Sxx + 1e-20)
    z = 1.96  # 95% CI（正态近似）
    ci = (b - z*se_b, b + z*se_b)
    return b, a, R2, ci


def plot_classwise_norms(args, hessian):
    """
    读取 hessian.file_dir 下的 classwise_norms.json，并绘制主图（scaling law 风格）：
      - 只画 Spectral：y = H_spectral，x = k+1
      - 坐标：x 轴 & y 轴都用对数刻度（幂律变线性）
      - 回归：对 (log x, log y) 做线性回归，回到原始坐标画幂律拟合：y = exp(a) * x^b
      - 理论线：
            config.mode == 'uniform'  画斜率 0 的水平幂律（常数）
            config.mode == '1k'       画斜率 -1 的幂律（~ 1/k）
            其他值则不画理论线
    保存到：{hessian.file_dir}/classwise_norms_plot.png
    """
    # 路径检查
    norms_path = os.path.join(hessian.file_dir, "classwise_norms.json")
    missing_hint = "请先运行 calc_classwise_hessian_spectrum() 生成 classwise_norms.json。"
    if not os.path.exists(norms_path):
        raise FileNotFoundError(f"{norms_path} 不存在，{missing_hint}")

    # 读取 JSON
    with open(norms_path, "r") as f:
        norms_dic = json.load(f)

    # 准备数据
    class_ids = sorted(int(k) for k in norms_dic.keys())
    k = np.array(class_ids, dtype=float)
    x_lin = k + 1.0  # x = k+1
    y_lin = np.array([float(norms_dic[str(c)]["spectral"]) for c in class_ids], dtype=float)

    # 处理非正值（log 需要）
    eps = 1e-12
    y_lin = np.maximum(y_lin, eps)

    # 在 log–log 空间做回归（幂律线性化）
    x_log = np.log(x_lin)
    y_log = np.log(y_lin)
    b, a, r2, ci = _linregress_np(x_log, y_log)   # y_log = a + b * x_log

    # 回到原始单位的拟合曲线：y = exp(a) * x^b
    A = np.exp(a)
    xgrid = np.logspace(np.log10(x_lin.min()), np.log10(x_lin.max()), 256)
    yfit = A * (xgrid ** b)

    # 理论线（原始单位）
    mode = getattr(args, "dataset_mode", None)
    y_theory = None
    theory_label = None
    if mode == "1k":
        # 期望：H ≈ C / (k+1)，估计 C = median(H * (k+1))
        C = np.median(y_lin * x_lin)
        y_theory = C / xgrid
        theory_label = "theory ~ 1/(k+1)"
    elif mode == "uniform":
        # 期望：H ≈ 常数，取中位数
        C = np.median(y_lin)
        y_theory = np.full_like(xgrid, C)
        theory_label = "theory ~ const"

    # 作图（log–log 轴；scatter + power-law fit + theory）
    plt.figure(figsize=(7.2, 5.0))
    plt.scatter(x_lin, y_lin, s=22, alpha=0.9, marker='^', label="Spectral (points)")
    if y_theory is not None:
        plt.plot(xgrid, y_theory, linewidth=1.8, label=theory_label)
    plt.plot(
        xgrid, yfit,
        linestyle="--", linewidth=2,
        label=f"power-law fit: slope={b:.2f} (95% CI [{ci[0]:.2f}, {ci[1]:.2f}]; R²={r2:.2f})"
    )

    # 轴&标题（scaling law 风格）
    plt.xscale("log")
    plt.yscale("log")  # 变更点：纵轴使用对数刻度
    plt.xlabel("class index  k+1  (log scale)")
    plt.ylabel("H_c (spectral)  (log scale)")
    title_suffix = f" | mode={mode}" if mode else ""
    plt.title(f"Classwise Hessian norm vs. class index (log–log){title_suffix}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(hessian.file_dir, "classwise_norms_plot.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Plot saved to {save_path}")


def build_loss_bins(loss_min, loss_max, num_bins=8):
    step = (loss_max - loss_min) / num_bins
    edges = [loss_max - i * step for i in range(num_bins)]
    edges.append(loss_min)
    return edges



def training(args, optimizer, scheduler, save_dir, train_loader, val_loader, model, criterion, writer, logger):
    print("Start Training ...")
    # Initialize SwanLab at the start of training
    if args.opt == 'sgd':
        comment = 'model-{}-opt-{}-lr-{}-momentum-{}-wd-{}-seed-{}-dsmode-{}'.format(
            args.arch, args.opt, args.lr, 
            args.momentum, args.wd, args.seed, 
            args.dataset_mode
        )
    else:
        comment = 'model-{}-opt-{}-lr-{}-beta1-{}-beta2-{}-wd-{}-seed-{}-dsmode-{}'.format(
            args.arch, args.opt, args.lr, 
            args.beta1, args.beta2, args.wd, 
            args.seed, args.dataset_mode
        )

    swanlab.init(
        project="HessianHetero_BalancedData_ViT",
        workspace="zhang_haoran",
        name=comment,
        config={
            "arch": args.arch,
            "optimizer": args.opt,
            "lr": args.lr,
            "batch_size": args.batchsize,
            "epochs": args.epochs,
            "scheduler": "CosineAnnealingLR_with_Warmup"  # 添加scheduler信息
        }
    )
    
    assert args.ckpt_std in ['epoch', 'loss']
    if args.ckpt_std == 'loss':
        loss_bin_edges = build_loss_bins(loss_min, loss_max)
        seen_loss_bins = set()
        print("loss_bin_edges:", loss_bin_edges)
    
    for epoch in range(args.start_epoch, args.epochs):
        # 移除原来的 adjust_learning_rate 调用
        # adjust_learning_rate(optimizer, epoch, args.lr)
        
        # 在每个epoch开始时获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch-{epoch}, Learning Rate: {current_lr:.6f}')
        print('task', save_dir)

        trainloss, trainacc = train(train_loader, model, criterion, optimizer, epoch, args.print_freq)
        testloss, testacc = test(val_loader, model)

        print("trainloss={} trainacc={} testloss={} testacc={}".format(trainloss, trainacc, testloss, testacc))
        
        # Log to SwanLab
        swanlab.log({
            "epoch": epoch,
            "lr": current_lr, 
            "train_loss": trainloss,
            "train_acc": trainacc,
            "val_loss": testloss,
            "val_acc": testacc
        })

        writer.add_scalar('trainloss', trainloss, epoch)
        writer.add_scalar('trainacc', trainacc, epoch)
        writer.add_scalar('testloss', testloss, epoch)
        writer.add_scalar('testacc', testacc, epoch)
        writer.add_scalar('learning_rate', current_lr, epoch)  # 添加学习率监控
        logger.append([epoch, trainloss, testloss, trainacc, testacc, current_lr])

        # 在每个epoch结束后更新学习率
        scheduler.step()

        if args.ckpt_std == 'loss':
            seen_loss_bins = save_checkpoint_on_loss_grid(
                    train_loss=trainloss, 
                    loss_bin_edges=loss_bin_edges, 
                    seen_bins=seen_loss_bins, 
                    epoch=epoch, 
                    model=model, 
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    args=args
            )


        # remember the best prec@1 and save checkpoint
        if args.ckpt_std == 'epoch': 
            if epoch % 10 == 1 or epoch == 1 or epoch == args.epochs - 1: 
                print('saving checkpoint..')
                save_completecheckpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()  # 保存scheduler状态
                }, savename=args.arch + args.opt + args.dataset_mode + '_ckpt_' + str(epoch))
    
    logger.close()
    logger.plot()
    io_utils.save_code(save_dir)
    yaml.safe_dump(args.__dict__, open(os.path.join(save_dir, 'config.yml'), 'w'), default_flow_style=False)
    


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    train_loss = 0
    correct = 0
    total = 0

    # switch to train mode
    model.train()

    end = time.time()

 
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        targets = targets.cuda()
        inputs = inputs.cuda()
        'regular update'
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        
        # Record grad norm
        log_weight_grad_norms(model)
        
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

#         progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    trainacc=correct/total
    trainloss=train_loss/(batch_idx+1)

    return trainloss, trainacc



def test(val_loader, model):
    
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
#             progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    testloss=test_loss/(batch_idx+1)
    testacc = correct/total

    return testloss, testacc





if __name__ == '__main__':
    main()
