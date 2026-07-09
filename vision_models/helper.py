import shutil
import torch
import os

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, savename='checkpoint.pth'):
    if not os.path.isdir('checkpoint'): # if no file named checkpoint, just create one
        os.mkdir('checkpoint')
    
    if is_best:
        torch.save(state, './checkpoint/'+str(savename)+'.pth')


def save_completecheckpoint(state, savename='checkpoint.pth'):
    if not os.path.isdir('checkpoint'):  # if no file named checkpoint, just create one
        os.mkdir('checkpoint')

    torch.save(state, './checkpoint/' + str(savename) + '.pth')
    
    
def get_loss_bin_index(loss_value, edges):
    """
    给定一个 loss 和 bin 边界（降序），返回这个 loss 落在哪个区间的 index。
    edges 长度为 B+1，返回的 index 在 [0, B-1] 之间。

    约定：
    - 区间定义为 [edges[i+1], edges[i])，i = 0..B-1
    - 如果 loss > edges[0]（高于最大值），返回 None：还没进入监控区间
    - 如果 loss <= edges[-1]（低于最小值），归到最后一个 bin（最好区间）
    """
    B = len(edges) - 1
    if B <= 0:
        return None

    # 高于最大值：还没进入我们定义的 loss grid，不触发任何 bin
    if loss_value > edges[0]:
        return None

    # 低于最小值：已经比预期还好，归到最后一个 bin
    if loss_value <= edges[-1]:
        return B - 1

    # 正常情况：找到 i 使得 edges[i] >= loss > edges[i+1]
    for i in range(B):
        if edges[i] >= loss_value > edges[i + 1]:
            return i

    # 理论上不该走到这里，兜底一下
    return None

    
    
def save_checkpoint_on_loss_grid(
    train_loss,
    loss_bin_edges,
    seen_bins,
    epoch,
    model,
    optimizer,
    scheduler,
    args,
):
    bin_idx = get_loss_bin_index(train_loss, loss_bin_edges)

    # 如果还没进入任何有效的 loss 区间（例如 loss > loss_max），直接返回
    if bin_idx is None:
        return seen_bins

    if bin_idx in seen_bins:
        return seen_bins

    high = loss_bin_edges[bin_idx]
    low = loss_bin_edges[bin_idx + 1]
    savename = f"{args.arch}{args.opt}{args.dataset_mode}_trainloss[{high:.4f}-{low:.4f})_bin{bin_idx}_epoch{epoch}"

    print(
        f"[save_checkpoint_on_loss_grid] train_loss={train_loss:.4f} "
        f"落入区间 [{low:.4f}, {high:.4f}) (bin={bin_idx})，保存 ckpt: {savename}"
    )

    save_completecheckpoint(
        {
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        savename=savename,
    )

    seen_bins.add(bin_idx)
    return seen_bins



def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))

    if hasattr(optimizer, 'param_groups'): 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # if hasattr(optimizer, 'lr'): 
    #     optimizer.lr = lr

    else: pass




def adjust_box(optimizer, epoch, init_box, init_boxtwo):


    
    box = init_box * (0.1 ** (epoch // 30))

    # print('iniital box', init_box)
    # print('epoch', epoch)
    # print('box',box)

    boxtwo = init_boxtwo * (0.1 ** (epoch // 30))

    if hasattr(optimizer, 'box_upperbound'): 
       optimizer.box_upperbound = box
    
    if hasattr(optimizer, 'box_upperbound_two'): 
       optimizer.box_upperbound_two = boxtwo
    else: pass

