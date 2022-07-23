# -*- coding:utf-8 -*-
import os, argparse
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from datasets.dataset import AugmentedDataset
from datasets.datasetting import get_data_setting
import models.networks as networks
# test for seg
""" set flags / seeds """
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description="")
parser.add_argument('--batch_size', type=int, default=1)
get_data_setting(parser, device_id=0)
parser.add_argument('--test_folds', default=[1])
parser.add_argument('--model_save_path', type=str,
                    default='../data/result/')
parser.add_argument('--model_save_name', type=str,default='model_G.ckpt')
parser.add_argument('--test_img_save_dir', type=str,
                    default=r'../data/result/test_imgs')
opt = parser.parse_args()

if __name__ == "__main__":
    """ datasets and dataloader """
    test_dataset = AugmentedDataset(opt.A_dir, opt.B_dir, opt.C_dir, opt.D_dir, opt.test_folds)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    """ instantiate network and loss function"""
    netG = networks.define_G(3, 1, 32, 'local_NDDR_v2',
                             n_downsample_global=4, n_blocks_global=9, n_local_enhancers=1,
                             n_blocks_local=3, norm='instance', gpu_ids=0)
    """ device configuration """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    """models init or load checkpoint"""
    model_save_path = os.path.join(opt.model_save_path, opt.model_save_name)
    netG.load_state_dict(torch.load(model_save_path)['netG'])

    if not os.path.exists(opt.test_img_save_dir): os.mkdir(opt.test_img_save_dir)
    imgdir = os.path.join(opt.test_img_save_dir, 'gen')
    if not os.path.exists(imgdir): os.mkdir(imgdir)

    netG.to(device)
    netG.eval()
    with torch.no_grad():
        for i, (real_A, real_B, _, _, imgname) in enumerate(test_loader):
            real_A = real_A.to(device)
            fake_B, _ = netG(real_A)


            img_path = os.path.join(opt.test_img_save_dir, 'gen', imgname[0][:-4] + '.png')
            save_image(fake_B, img_path)
