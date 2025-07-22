from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import us_dataset, trans
from torchvision import transforms
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from natsort import natsorted
from models.model import CONFIGS as CONFIGS_TM
from models.model import PADReg
from pytorch_msssim import ssim, ms_ssim, SSIM
from torch.optim.lr_scheduler import StepLR
import random
import argparse
import time
import torch.nn.functional as F

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=float, help='weight of similarity loss', default=1)
    parser.add_argument('-r', type=float, help='weight of regulization loss', default=0.03)
    parser.add_argument('-lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('-b', type=int, help='batch_size', default=32)
    parser.add_argument('-img_size', type=int, help='image size', default=256)

    # 解析参数
    args = parser.parse_args()
    return args

def seed_torch(seed = 1000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def _init_fn(worker_id):
    np.random.seed(int(1000)+worker_id)

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    seed_torch()
    args = parse_arguments()
    """Training Parameters"""
    weights = [args.m,args.r] 
    lr = args.lr 
    batch_size = args.b
    sch_step_size = 30
    sch_gamma = 0.7


    """Modify the file name"""
    save_dir = f'Experiment_{time.strftime("%Y%m%d-%H%M%S")}_MSE_{weights[0]}_rg_{weights[1]}_lr_{lr}/'
    log_dir = 'runs/'
    max_epoch = 100
    cont_training = False 

    """Dataset Settings"""
    data_list_train = 'PADReg/data/video/input_train_fg.txt'
    data_list_val = 'PADReg/data/video/input_valid_fg.txt'
    prefix = 'PADReg/data/video/'

    force_folder = 'PADReg/data/force'
    ann_folder_train = None
    ann_folder_valid = 'PADReg/data/annotations'

    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists(log_dir+save_dir):
        os.makedirs(log_dir+save_dir)
    sys.stdout = Logger(log_dir+save_dir)
    epoch_start = 0
    print(f"lr:{lr}, batch_size:{batch_size}")
    print(save_dir)

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    model = PADReg(config)
    model.cuda()


    '''
    If continue from previous training
    '''
    if cont_training:
        model_dir = 'experiments/'
        updated_lr = lr
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model loaded!')
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.Resize_img([args.img_size,args.img_size]),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])
    valid_composed = transforms.Compose([trans.Resize_img([args.img_size,args.img_size]),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])
    train_set = us_dataset.USDataset(data_list_train, prefix=prefix,
                                     force_folder=force_folder,
                                     transforms=train_composed,
                                     ann_folder=ann_folder_train)
    val_set = us_dataset.USInferDataset(data_list_val, prefix=prefix, 
                                        force_folder=force_folder, 
                                        transforms=valid_composed,
                                        ann_folder=ann_folder_valid)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,worker_init_fn=_init_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,worker_init_fn=_init_fn)
    print(f"{len(train_set)} training pairs are loaded.")
    print(f"{len(val_set)} training pairs are loaded.")

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    scheduler = StepLR(optimizer, step_size=sch_step_size, gamma=sch_gamma) 

    """Define losses for training"""
    criterion_sim = losses.MSE()
    criterion_reg = losses.Grad('l2')

    """Define losses for evaluation"""
    criterion_seg = losses.Dice()
    ssim = SSIM(data_range=1, size_average=True, channel=1)
    mse = losses.MSE()
    mi = losses.MutualInformation()
    dcy_rate = losses.DiscrepancyRate()
    hd95 = losses.HD95()


    best_ncc = 0
    writer = SummaryWriter(log_dir=log_dir+save_dir)
    print('Training Starts')
    for epoch in range(epoch_start, max_epoch):
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            f = data[2]
            x_in = torch.cat((x,y), dim=1)
            output = model(x_in,f)

            l_sim = criterion_sim(output[0], y) * weights[0]
            l_reg = criterion_reg(output[1], y) * weights[1]
            loss = l_sim + l_reg
           
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output
            # flip fixed and moving images
            f_flip = torch.flip(f,dims=[1])
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in,f_flip)
  

            # calculate loss
            l_sim_r = criterion_sim(output[0], x) * weights[0]
            l_reg_r = criterion_reg(output[1], x) * weights[1]
            loss = l_sim_r + l_reg_r
            l_sim += l_sim_r
            l_reg += l_reg_r
           

            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img MSE: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), l_sim.item() / 2, l_reg.item() / 2))
            writer.add_scalar('Loss/train', loss.item(), idx + len(train_loader) * epoch)
            writer.add_scalar('Loss/train_mse', l_sim.item() / 2, idx + len(train_loader) * epoch)
            writer.add_scalar('Loss/train_reg', l_reg.item() / 2, idx + len(train_loader) * epoch)
            
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch)

        '''
        Validation
        '''
        eval_ncc = utils.AverageMeter() # SSIM
        eval_ncc_2 = utils.AverageMeter() # MSE
        eval_ncc_4 = utils.AverageMeter() # MI
        eval_ncc_5 = utils.AverageMeter() # Dice loss
        eval_ncc_6 = utils.AverageMeter() # DCY Rate
        eval_ncc_7 = utils.AverageMeter() # HD95

        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[2]
                y = data[3]
                f = data[4]
                if len(data) > 5:
                    m_ann = data[5]
                    f_ann = data[6]
                else:
                    m_ann = None
                    f_ann = None
                f_flip = torch.flip(f,dims=[1])

                x_in = torch.cat((y, x), dim=1)
                output = model(x_in,f_flip,f_ann)

                ncc = ssim(output[0], x)
                eval_ncc.update(ncc.item(), x.numel())
                ncc_2 = mse(output[0],x)
                eval_ncc_2.update(ncc_2.item(), x.numel())
                ncc_4 = mi(x,output[0])
                eval_ncc_4.update(ncc_4.item(), x.numel())
                ncc_6 = dcy_rate(output[1],f_flip)
                eval_ncc_6.update(ncc_6.item(),x.numel())

                if len(output) == 4:
                    ncc_5 = criterion_seg(output[3].to(torch.int64),m_ann)
                    eval_ncc_5.update(ncc_5.item(), x.numel())
                    ncc_7 = hd95(output[3].to(torch.int64),m_ann)
                    eval_ncc_7.update(ncc_7.item(),x.numel())


                #flip image
                x_in = torch.cat((x, y), dim=1)
                output = model(x_in,f,m_ann)  

                ncc = ssim(output[0], y)
                eval_ncc.update(ncc.item(), y.numel())
                ncc_2 = mse(output[0],y)
                eval_ncc_2.update(ncc_2.item(), y.numel())
                ncc_4 = mi(y,output[0])
                eval_ncc_4.update(ncc_4.item(), y.numel())
                ncc_6 = dcy_rate(output[1],f)
                eval_ncc_6.update(ncc_6.item(),y.numel())
                if len(output) == 4:
                    ncc_5 = criterion_seg(output[3].to(torch.int64),f_ann)
                    eval_ncc_5.update(ncc_5.item(), y.numel())
                    ncc_7 = hd95(output[3].to(torch.int64),f_ann)
                    eval_ncc_7.update(ncc_7.item(),y.numel())

        scheduler.step()
        print("eval_dice:",eval_ncc_5.avg)
        best_ncc = max(eval_ncc_5.avg, best_ncc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_ncc': best_ncc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/'+save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_ncc_5.avg))
        writer.add_scalar('Val/SSIM', eval_ncc.avg, epoch)
        writer.add_scalar('Val/MSE', eval_ncc_2.avg, epoch)
        writer.add_scalar('Val/MI', eval_ncc_4.avg, epoch)
        writer.add_scalar('Val/DCY', eval_ncc_6.avg, epoch)
        if len(output) == 4:
            writer.add_scalar('Val/DICE', eval_ncc_5.avg, epoch)
            writer.add_scalar('Val/HD95', eval_ncc_7.avg, epoch)

        loss_all.reset()

    writer.close()


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))
    torch.save(state, save_dir+filename)


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
