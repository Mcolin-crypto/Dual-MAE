import time
from config import load_args
from hsidata_perclass import *
import torch
from torch.utils.data import DataLoader
from models.model_mae import *
from models.model_vit import *
from functools import partial
from engine import *
from utils import *
from timm.models.layers import trunc_normal_
from pathlib import Path
from timm.scheduler.cosine_lr import CosineLRScheduler
from logger import create_logger

args = load_args()

device = torch.device('cuda')
datasets = args.datasets
_,gt_path,nbands,nclass = load_data(datasets)
nclass = nclass.astype(np.int64)

class two_branch_model(nn.Module):
    def __init__(self, spemodel, spamodel, nclass):
        super(two_branch_model, self).__init__()
        self.spemodel = spemodel
        self.spamodel = spamodel
        self.cls_mlp = nn.Sequential(
                    nn.Linear(args.feature_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, nclass),
                )
        self.fc = nn.Linear(args.feature_dim * 2, nclass)
        self.add_module('fusion', branch_fusion(args.feature_dim * 2))
        
    def forward(self, x1, x2):
        x_spe = self.spemodel(x1)
        fe_e = x_spe
        pred_spe = self.cls_mlp(x_spe)
        x_spa = self.spamodel(x2)
        
        pred_spa = self.cls_mlp(x_spa)
        x = torch.hstack([x_spe, x_spa])
        x = self.fusion(x)
        out = self.fc(x)
        
        return out,pred_spe,pred_spa,fe_e

if __name__ == '__main__':
       
    # prepare directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # logger
    name = 'dual-mae' +'_'+ datasets
    logger = create_logger(output_dir=log_dir, name=name)
    logger.info(args)
    
    acc = np.zeros([args.nexp, 1])
    A = np.zeros([args.nexp, nclass])
    kappa = np.zeros([args.nexp, 1])
    exp = 1
    start_time = time.time()
    gt = {k: v for k, v in sio.loadmat(gt_path).items()
                   if isinstance(v, np.ndarray) and 'map' not in k}    
    gt = list(gt.values())[0]
    n = np.zeros(nclass,dtype=np.int64)
    ntrain_perclass = np.ones(nclass,dtype=np.int64) 
    for i in range(nclass):
        coord_x,coord_y = np.where(gt == i + 1)
        n[i] = len(coord_x)
        
        ntrain_perclass[i] = n[i] * args.ratio_train
        if ntrain_perclass[i] <= 15:
            ntrain_perclass[i] = 15
        
        logger.info(f"class{i}_num:{n[i]}")
        
    for i in range(nclass):
        logger.info(f"class{i}_ntrain:{ntrain_perclass[i]}")
     
    for i in range (args.nexp):
        
        cur_seed = np.random.randint(0,1000)
        set_seed(cur_seed)
        logger.info(f'EXP={exp}:Seed={cur_seed}')
        train_set = get_datasets(datasets,'train',cur_seed)
        test_set = get_datasets(datasets,'test',cur_seed)
        ntrain = len(train_set)
        ntest = len(test_set)
        logger.info(f'train_num:{ntrain}')
        logger.info(f'test_num:{ntest}')

        train_loader = DataLoader(train_set, args.batch_size, drop_last=False, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_set, args.batch_size, drop_last=False, shuffle=True, pin_memory=True)

        spamodel = spa_mae(img_size=(args.windowsize,args.windowsize), in_chans=args.c_pca,
                            hid_chans=args.hid_chans,
                            embed_dim=args.embed_dim_a, 
                            depth=args.encoder_depth, 
                            num_heads=args.num_heads,
                            decoder_embed_dim=args.decoder_embed_dim,
                            decoder_depth=args.decoder_embed_dim,
                            decoder_num_heads=args.decoder_num_heads,
                            patch_size=3, 
                            norm_layer=partial(nn.LayerNorm, eps=1e-6),
                            nb_classes=nclass,)
        
        spemodel = spe_mae(img_size=(1,1),
                            patch_size=1,
                            in_chans=nbands,
                            embed_dim=args.embed_dim_e, 
                            depth=args.encoder_depth, 
                            num_heads=args.num_heads,
                            decoder_embed_dim=args.decoder_embed_dim, 
                            decoder_depth=args.decoder_depth,
                            decoder_num_heads=args.decoder_num_heads,
                            nb_classes=nclass,)
        
        model = two_branch_model(spemodel,spamodel,nclass)
        model.to(device)
        
        # ##pretrain
        # params_dict = [{"params": [p for p in model.spamodel.parameters() if spamodel], "lr": args.lr}, 
        #             {"params": [p for p in model.spemodel.parameters() if spemodel], "lr": 5e-4}]
        # optimizer = torch.optim.Adam(params_dict , weight_decay=1e-4)
        # # logger.info(optimizer)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
        
        # logger.info(f"Start training for {args.epochs} epochs")
        
        # for epoch in range (150):
        #     pretrain(model,train_loader,optimizer,scheduler,epoch,150,logger=logger)
        # state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        
        # torch.save(state,'pretrain_checkpoint/checkpoint.pt')
        # logger.info('pretrain checkpoint saved!')
        
        ## finetune
        spemodel_v = spe_vit(img_size=(1,1),
                            in_chans=nbands,
                            patch_size=1,
                            embed_dim=128, 
                            depth=4, 
                            num_heads=8,
                            num_classes=args.feature_dim,
                            )
        spamodel_v = spa_vit(img_size=args.windowsize,
                            in_chans=args.c_pca, 
                            hid_chans= args.hid_chans,
                            embed_dim=128, 
                            depth=4,
                            num_heads=8, 
                            patch_size=3,
                            num_classes=args.feature_dim,
                            )
        model_fine = two_branch_model(spemodel_v,spamodel_v,nclass)
        model_fine.to(device)
        
        ##加载预训练的权重
        checkpoint = torch.load('pretrain_checkpoint/checkpoint.pt')
        logger.info("Load pre-trained checkpoint from:'pretrain_checkpoint/checkpoint.pt'")
        checkpoint_model = checkpoint['model']
        state_dict = model_fine.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                
        for key in ['fc.0.weight', 'fusion.fc1.weight', 'fusion.fc2.weight']:
            if key in checkpoint_model and checkpoint_model[key].shape != state_dict[key].shape:
                logger.info(f"Removing key {key} from pretrained checkpoint")
                del checkpoint_model[key]
                
        # interpolate position embedding
        interpolate_pos_embed(model_fine, checkpoint_model)

        # load pre-trained model
        msg = model_fine.load_state_dict(checkpoint_model, strict=False)
        trunc_normal_(model_fine.spamodel.head.weight, std=2e-5)
        trunc_normal_(model_fine.spemodel.head.weight, std=2e-5)
        
        params_dict = [{"params": [p for p in model_fine.spamodel.parameters() if spamodel_v], "lr": args.lr}, 
                    {"params": [p for p in model_fine.spemodel.parameters() if spemodel_v], "lr": 5e-4}]
        optimizer = torch.optim.Adam(params_dict , weight_decay=1e-4) 
        scheduler = CosineLRScheduler(
                optimizer,
                t_initial=args.epochs*len(train_loader),
                lr_min=1e-6,
                warmup_lr_init=1e-5,
                warmup_t = 0,
                cycle_limit=1,
                t_in_epochs=False,
            )
        
        if args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        # ##train
        best_acc = 0.0
        best_model_acc = 0.0
        for epoch in range(args.epochs):        
            model_fine = train(model_fine,train_loader,criterion,optimizer,scheduler,epoch,args.epochs,logger=logger,)
            if epoch % 2 == 0 and (epoch + 1) > (args.epochs - 20):
                
                val_acc = test(model_fine, test_loader, criterion, acc, A, kappa, exp, logger=logger)[0]
                save_checkpoint(output_dir=args.output_dir, epoch=epoch, model=model_fine, optimizer=optimizer, lr_scheduler=scheduler, max_acc=best_acc, val_acc=val_acc, logger=logger)
                
                best_acc = max(best_acc, val_acc)
                
        train_end = time.time()
        
        ##test
        test_acc, _ , acc, A, kappa = test(model_fine, test_loader, criterion, acc, A, kappa, exp, logger=logger)
        save_checkpoint(model=model_fine, save_path=Path(args.output_dir) / f'last_model_{test_acc:.4f}.pth')
        logger.info(f'Last model accuracy: {test_acc:.4f}')
        
        model_fine.load_state_dict(torch.load(Path(args.output_dir) / 'best_ckpt.pth')['model'])
        test_acc, _, acc, A, kappa= test(model_fine, test_loader , criterion, acc, A, kappa, exp, logger=logger)
        test_end = time.time()
        logger.info("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))

        save_checkpoint(model=model_fine, save_path=Path(args.output_dir) / f'best_model_{test_acc:.4f}.pth') 
        logger.info(f'Best model accuracy: {test_acc:.4f}')
        
        exp += 1
        
    AA = np.mean(A, 1)
    AAMean = np.mean(AA, 0)
    AAStd = np.std(AA)
    AMean = np.mean(A, 0)
    AStd = np.std(A, 0)
    OAMean = np.mean(acc)
    OAStd = np.std(acc)
    kMean = np.mean(kappa)
    kStd = np.std(kappa)
    
    logger.info("average OA: " + "{:.2f}".format(OAMean) + " ± " + "{:.2f}".format(OAStd))
    logger.info("average AA: " + "{:.2f}".format(100 * AAMean) + " ± " + "{:.2f}".format(100 * AAStd))
    logger.info("average kappa: " + "{:.4f}".format(100 * kMean) + " ± " + "{:.4f}".format(100 * kStd))
    
    for i in range(nclass):
        logger.info("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " ± " + "{:.2f}".format(100 * AStd[i]))
        
    total_time = time.time() - start_time
    logger.info(f'total time: {total_time}')