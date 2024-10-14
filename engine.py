##with klDivLoss
from torch import nn
import torch
from torch.utils.data.dataloader import DataLoader
from timm.scheduler.scheduler import Scheduler
from timm.utils import  AverageMeter, accuracy
import time
import datetime
import numpy as np
from sklearn import metrics
from config import load_args
from logger import false_logger
import torch.nn.functional as F

args = load_args()

def pretrain(model:torch.nn.Module,
             data_loader:DataLoader,
            #  criterion:torch.nn.Module,
             optimizer:torch.optim.Optimizer,
             scheduler:Scheduler,
             epoch=None,
             epochs=None,
             logger=false_logger
             ):
    model.train()
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()
    accum_iter = 1
    loss_meter = AverageMeter()
    lr_meter = AverageMeter()
    print_freq = 100
    num_steps = len(data_loader)
    start_time = time.time()
    for idx, (samples,spesamples,targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        spesamples = spesamples.cuda(non_blocking =True)
        y = targets.cuda(non_blocking = True)

        loss1,_,_,logits1= model.spemodel(spesamples,y,mask_ratio=0.3)
        cls_loss1 = criterion(logits1 / 1.0, y) * args.cls_loss_ratio
        loss1 = loss1 + cls_loss1
        
        loss2,_,_,logits2 = model.spamodel(samples,y,mask_ratio=args.mask_ratio)
        cls_loss2 = criterion(logits2 / 1.0, y) * args.cls_loss_ratio
        loss2 = loss2 + cls_loss2
        
        loss = loss1 + loss2
        
        loss /= accum_iter
        
        loss.backward()
        optimizer.step()
        
        if (idx + 1) % accum_iter == 0:
            scheduler.step()
            optimizer.zero_grad()
        loss_meter.update(loss.item())
        
        lr = optimizer.param_groups[0]["lr"]
        lr_meter.update(lr)
        
        if (idx + 1) % print_freq == 0 or (idx + 1 ) == num_steps:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train:[{epoch + 1}/{epochs}][{idx + 1}/{num_steps}]\t'
                f'lr {lr:.8f}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                )
    epoch_time = time.time() - start_time 
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time))}")
   
def train(model: torch.nn.Module,
        train_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Scheduler,
        epoch=None,
        epochs=None,
        logger=false_logger,
        ):
    
    model.train()
    criterion.train()
        
    num_steps =  len(train_loader)
    batch_time = AverageMeter()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    klloss = nn.KLDivLoss(reduction = 'batchmean')
    
    temp = 1.0

    PRINT_FREQ = 100
    
    start = time.time()
    end = time.time()
    
    for idx, (samples,spesamples,targets ) in enumerate(train_loader):
        samples = samples.cuda(non_blocking=True)
        spesamples = spesamples.cuda(non_blocking=True)
        gt = targets.cuda(non_blocking=True)
        output,pred_e,pred_a,_=model(spesamples,samples)

        loss_kl = klloss(F.log_softmax(pred_e / temp, dim=1), F.softmax(pred_a / temp, dim=1))
        loss = criterion(output, gt) 
        loss += loss_kl * args.kl_ratio

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step_update(epoch * num_steps + idx)
            
        train_loss_meter.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        train_acc_meter.update(accuracy(output, gt, topk=(1,))[0])
            
        if (idx + 1) % PRINT_FREQ == 0 or (idx + 1 ) == num_steps:
            lr = optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train:[{epoch}/{epochs}][{idx + 1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} \t'
                f'lr {lr:.8f}\t'
                # f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {train_loss_meter.val:.4f} ({train_loss_meter.avg:.4f})\t'
                f'acc {train_acc_meter.val:.4f} ({train_acc_meter.avg:.4f})\t'
                )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return model

@torch.no_grad()
def test(model: torch.nn.Module,
        data_loader: DataLoader,
        criterion: torch.nn.Module,
        acc, A, kappa, exp_time,
        logger=false_logger):
    model.eval()
    criterion.eval()
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
        
    test_pred_all = []
    test_all = []
    correct = 0

    for idx, (samples,spesamples,targets ) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        spesamples = spesamples.cuda(non_blocking=True)
        gt = targets.cuda(non_blocking=True)
        output,_,_,_=model(spesamples,samples)
        loss = criterion(output, gt)
        _, predicted = torch.max(output, 1)
        test_all = np.concatenate([test_all, gt.cpu().numpy()])
        test_pred_all = np.concatenate([test_pred_all, predicted.cpu()])
        correct += predicted.eq(gt.view_as(predicted)).cpu().sum()

        acc2 = accuracy(output, gt, topk=(1,))[0]
        end = time.time()
        loss_meter.update(loss.item())
        acc_meter.update(acc2.item())
        batch_time.update(time.time() - end)

        if (idx + 1) % 2000 == 0 or (idx + 1) == num_steps:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'EVAL: [{(idx + 1)}/{len(data_loader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    correct_rate = 100. * correct.item() / len(data_loader.dataset)  # 将 Tensor 转换为 Python 标量
    if(acc[exp_time-1] < correct_rate): 
        acc[exp_time-1] = 100. * correct / len(data_loader.dataset)
        C = metrics.confusion_matrix(test_all, test_pred_all)
        A[exp_time-1, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)
        kappa[exp_time-1] = metrics.cohen_kappa_score(test_all, test_pred_all)
    logger.info(f'EVAL * Acc@ {acc_meter.avg:.3f}')
    
    return acc_meter.avg, loss_meter.avg, acc, A, kappa
   
def tr_acc(model, data_loader,criterion):
    train_loader = data_loader
    train_loss = 0
    corr_num = 0
    sum_image = 0
    for idx, (image_batch, label_batch) in enumerate(train_loader):        
        trans_image_batch = image_batch.cuda()
        label_batch = label_batch.cuda()
        logits = model(trans_image_batch)
        if isinstance(logits,tuple):
            logits = logits[-1]
        pred = torch.max(logits, dim=1)[1]
        loss = criterion(logits, label_batch)                
        train_loss = train_loss + loss.cpu().data.numpy()
        corr_num = torch.eq(pred, label_batch).float().sum().cpu().numpy() + corr_num
        sum_image += 1   
    return corr_num/sum_image, train_loss/(idx+1) 