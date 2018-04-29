import os
import torch
from torch.autograd import Variable

def save_checkpoint(state, is_best, model_name, filename="model_ckp", bestchkp="model_ckp_best"):
    import shutil
    """Saves checkpoint to disk"""
    directory = f"checkpoint/{model_name}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{directory}/{bestchkp}')


def load_checkpoint(checkpoint):
    if os.path.isfile(checkpoint):
        print("=> loading checkpoint '{}'".format(checkpoint))
        ckp = torch.load(checkpoint)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint, checkpoint['epoch']))
        return ckp
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint))
        return None


def to_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def train_triplet(epoch, train_loader, model, criterion, optimizer):
    import torch.nn.functional as F
    from tqdm import tqdm

    losses = AverageMeter()
    # emb_norms = AverageMeter()

    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data_a, data_n, data_p) in pbar:
        model.zero_grad()
        # batch size
        batch_size = data_a.size(0)
        # if config.USE_GPU:
        data_a, data_n, data_p = to_cuda(data_a), to_cuda(data_n), to_cuda(data_p)
        data_a, data_n, data_p = Variable(data_a), Variable(data_n), Variable(data_p)

        # compute embed
        embedded_a = model(data_a)
        embedded_p = model(data_p)
        embedded_n = model(data_n)

        loss = criterion(embedded_a, embedded_p, embedded_n)

        # measure accuracy and record loss
        losses.update(loss.data[0], batch_size)

        # compute gradient and do optimizer step
        loss.backward()
        optimizer.step()

        pbar.set_description("Epoch train {}, "
                             "Loss {:.4f} ({:.4f}), ".format(
            epoch, losses.val, losses.avg
        ))


def test_triplet(epoch, test_loader, model, criterion):
    import torch.nn.functional as F
    from tqdm import tqdm
    losses = AverageMeter()
    # switch to evaluation mode
    model.eval()
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for batch_idx, (data_a, data_n, data_p) in pbar:
        # batch size
        batch_size = data_a.size(0)
        # if config.USE_GPU:
        data_a, data_n, data_p = to_cuda(data_a), to_cuda(data_n), to_cuda(data_p)
        data_a, data_n, data_p = Variable(data_a), Variable(data_n), Variable(data_p)

        # compute embed
        embedded_a = model(data_a)
        embedded_p = model(data_p)
        embedded_n = model(data_n)

        loss = criterion(embedded_a, embedded_p, embedded_n)

        # measure accuracy and record loss
        losses.update(loss.data[0], batch_size)
        pbar.set_description("Epoch test {}, "
                             "Loss {:.4f} ({:.4f}) ".format(
            epoch, losses.val, losses.avg
        ))

    return losses.avg


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


def accuracy(dista, distb, margin=0.2):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]
