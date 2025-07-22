import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import us_dataset, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
from models.model import PADReg
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.nn as nn
import logging
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, help='the path of checkpoints')
    parser.add_argument('-img_size', type=int, help='image size', default=256)

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    # Modify the test data list here
    data_list_test = 'PADReg/data/video/input_valid_fg.txt'
    prefix = 'PADReg/data/video/'
    force_folder = 'PADReg/data/force'
    ann_folder_valid = 'PADReg/data/annotations'
    model_idx = -1

    # Modify parameters here
    model_folder = args.c

    result_dir = 'Quantitative_Results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    model_dir = 'experiments/' + model_folder
    img_dir = os.path.join(result_dir,model_folder+'images')
    csv_dir = os.path.join(result_dir,model_folder,model_folder[:-1]+'.csv')

    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+model_folder[:-1]+'.csv'):
        os.remove('Quantitative_Results/'+model_folder[:-1]+'.csv')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir) 

    log_file = 'log.txt'
    log_path = os.path.join(result_dir,model_folder,log_file)
    logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),  # 将日志写入文件
        logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    logger = logging.getLogger()

    config = CONFIGS_TM['TransMorph-Sin']
    model = PADReg(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    
    test_composed = transforms.Compose([trans.Resize_img([args.img_size,args.img_size]), 
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])
    test_set = us_dataset.USInferDataset(data_list_test, prefix=prefix, 
                                        force_folder=force_folder, 
                                        transforms=test_composed,
                                        ann_folder=ann_folder_valid)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    criterion_seg = losses.Dice()
    ssim = SSIM(data_range=1, size_average=True, channel=1)
    mse = losses.MSE()
    mi = losses.MutualInformation()
    dcy_rate = losses.DiscrepancyRate()
    hd95 = losses.HD95()

    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()

    eval_ncc_2 = utils.AverageMeter() # MSE
    eval_ncc_4 = utils.AverageMeter() # MI
    eval_ncc_5 = utils.AverageMeter() # Dice loss
    eval_ncc_6 = utils.AverageMeter() # DCY Rate
    eval_ncc_7 = utils.AverageMeter() # HD95

    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[2]
            y = data[3]
            f = data[4]
            f_flip = torch.flip(f, dims=[1])
            if len(data) > 5:
                m_ann = data[5]
                f_ann = data[6]
            else:
                m_ann = None
                f_ann = None

            # flip image
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in,f_flip,f_ann)

            """Cal and record metrics"""
            ncc = criterion_seg(m_ann, f_ann)
            eval_dsc_raw.update(ncc.item(), x.numel())
            ncc = ssim(output[0], x)
            eval_dsc_def.update(ncc.item(), x.numel())

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

        
            # ori image
            x_in = torch.cat((x, y), dim=1)
            output = model(x_in,f,m_ann)


            """Cal and record metrics"""
            ncc = ssim(y, x)
            eval_dsc_raw.update(ncc.item(), x.numel())
            ncc = ssim(output[0], y)
            eval_dsc_def.update(ncc.item(), y.numel())

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

            stdy_idx += 1

            print(f'Processed {stdy_idx}/{len(test_loader)} image pairs')

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        logging.info(f'Val/SSIM:{eval_dsc_def.avg}')
        logging.info(f'Val/MSE:{eval_ncc_2.avg}')
        logging.info(f'Val/MI:{eval_ncc_4.avg}')

        logging.info(f'Val/DCY:{eval_ncc_6.avg}')
        if len(output) == 4:
            logging.info(f'Val/DICE:{eval_ncc_5.avg}')
            logging.info(f'Val/HD95:{eval_ncc_7.avg}')

def csv_writter(line, name):
    with open(name, 'a') as file:
        file.write(line)
        file.write('\n')


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