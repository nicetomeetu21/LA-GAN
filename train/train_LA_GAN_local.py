# -*- coding:utf-8 -*-
import argparse
import os
import shutil
import sys

import albumentations as A
import lpips
import numpy as np
import torch
import torch.nn as nn
import tqdm
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import models.networks as networks
from datasets import datasetting
from datasets.dataset import AugmentedDataset
from models.dice_loss1 import DiceLoss
from utils.log_function import print_options, print_network
from utils.scheduler import get_scheduler
from utils.util import set_requires_grad

""" set flags / seeds """
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class DEGAN(nn.Module):
    def __init__(self, opt):
        super(DEGAN, self).__init__()
        self.opt = opt
        self.netG = networks.define_G(3, 1, 32, 'local_NDDR_v2',
                                      n_downsample_global=4, n_blocks_global=9, n_local_enhancers=1,
                                      n_blocks_local=3, norm='instance', gpu_ids=0)
        model_save_path = ''
        self.load_network(self.netG, model_save_path)
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')

        self.netD_nm = networks.define_D(2, 64, 3, 'instance', False, 2, True, gpu_ids=0)
        self.netD_abnm = networks.define_D(2, 64, 3, 'instance', False, 2, True, gpu_ids=0)

        """ optimizer and scheduler """
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_g, betas=(0.5, 0.999))
        self.optimizer_D_nm = torch.optim.Adam(self.netD_nm.parameters(), lr=opt.lr_d, betas=(0.5, 0.999))
        self.optimizer_D_abnm = torch.optim.Adam(self.netD_abnm.parameters(), lr=opt.lr_d, betas=(0.5, 0.999))

        self.scheduler_G = get_scheduler(self.optimizer_G, opt.schedule_times, 0,
                                         opt.schedule_times - opt.schedule_decay_times)
        self.scheduler_D_nm = get_scheduler(self.optimizer_D_nm, opt.schedule_times, 0,
                                            opt.schedule_times - opt.schedule_decay_times)
        self.scheduler_D_abnm = get_scheduler(self.optimizer_D_abnm, opt.schedule_times, 0,
                                              opt.schedule_times - opt.schedule_decay_times)
        self.schedule_iters = opt.num_iters // opt.schedule_times
        print('scheduler total %d iters from %d iters, iter_size %d' % (
            opt.schedule_times, opt.schedule_times - opt.schedule_decay_times, self.schedule_iters))

        # define loss functions
        self.criterionGAN = networks.GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionVGG = networks.VGGLoss()

        self.criterionDice = DiceLoss()
        self.criterionCE = torch.nn.BCEWithLogitsLoss()

    def load_network(self, network, save_path=''):
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            raise ('Generator must exist!')
        else:
            # network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path)['netG'])
            except:
                pretrained_dict = torch.load(save_path)['netG']
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    print(
                        'Pretrained network has excessive layers; Only loading layers that are used')
                except:
                    print(
                        'Pretrained network has fewer layers; The following are not initialized:')
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def criterion_S(self, seg_out, seg_label):
        loss_Dice = self.criterionDice(seg_out, seg_label)
        loss_CE = self.criterionCE(seg_out, seg_label)
        loss_S = (loss_Dice + loss_CE) * 0.5
        writer.add_scalars('losses_S', {'loss_S': loss_S, 'loss_CE': loss_CE, 'loss_Dice': loss_Dice},
                           global_step=global_iter)
        return loss_S

    def criterion_PD_singleScale(self, img, mask, netD, target_is_real, for_discriminator):
        _, _, h, w = img.size()
        weight = (torch.sum(mask) / (h * w))
        D_in = torch.cat([img, mask], dim=1)
        if for_discriminator:
            D_in = D_in.detach()
            weight = weight.detach()
        pred = netD(D_in)
        loss_D = self.criterionGAN(pred, target_is_real=target_is_real) * weight
        return loss_D

    def criterion_PD_singleMask(self, img, mask, netD, target_is_real, for_discriminator):
        n, _, h, w = img.size()
        loss_PD = 0
        step = h
        for splits in range(3):
            loss_PD1 = 0
            cnt = 0
            for x in range(0, h, step):
                for y in range(0, w, step):
                    mask_region = mask[:, :, x:x + step, y:y + step]
                    if torch.sum(mask_region) > 0:
                        img_region = img[:, :, x:x + step, y:y + step]
                        loss_PD1 += self.criterion_PD_singleScale(img_region, mask_region, netD,
                                                                  target_is_real=target_is_real,
                                                                  for_discriminator=for_discriminator)
                        cnt += 1
            loss_PD1 /= cnt
            loss_PD += loss_PD1
            step //= 2
        return loss_PD

    def criterion_PD(self, fake_B, real_B, mask, netD):
        loss_D_fake = self.criterion_PD_singleMask(fake_B, mask, netD, target_is_real=False, for_discriminator=True)
        loss_D_real = self.criterion_PD_singleMask(real_B, mask, netD, target_is_real=True, for_discriminator=True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    def criterion_GAN_FM_VGG(self, fake_B, real_B, real_A, nm_mask, abnm_mask):
        fake_in = torch.cat([fake_B, nm_mask], dim=1)
        pred_fake = self.netD_nm(fake_in)
        real_in = torch.cat([real_B, nm_mask], dim=1)
        pred_real = self.netD_nm(real_in)
        loss_G_GAN_Feat = 0
        feat_weights = 4.0 / (3 + 1)
        D_weights = 1.0 / 2
        for i in range(2):
            for j in range(len(pred_fake[i]) - 1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                                   self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * 10

        fake_in = torch.cat([fake_B, abnm_mask], dim=1)
        pred_fake = self.netD_abnm(fake_in)
        real_in = torch.cat([real_B, abnm_mask], dim=1)
        pred_real = self.netD_abnm(real_in)
        feat_weights = 4.0 / (3 + 1)
        D_weights = 1.0 / 2
        for i in range(2):
            for j in range(len(pred_fake[i]) - 1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                                   self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * 10

        # VGG feature matching loss
        loss_G_VGG = self.criterionVGG(torch.cat([fake_B, fake_B, fake_B], dim=1),
                                       torch.cat([real_B, real_B, real_B], dim=1)) * 10

        loss_G_GAN = 0
        loss_G = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG

        writer.add_scalars('losses_G_HD',
                           {'loss_G_GAN': loss_G_GAN, 'loss_G_GAN_Feat': loss_G_GAN_Feat, 'loss_G_VGG': loss_G_VGG},
                           global_step=global_iter)
        return loss_G

    def criterion_G(self, fake_B, real_B, real_A, nm_mask, abnm_mask):
        loss_G_HD = self.criterion_GAN_FM_VGG(fake_B, real_B, real_A, nm_mask, abnm_mask)

        loss_G_RGAN = 0
        if torch.sum(abnm_mask) > 0:
            loss_G_PG_abnm = self.criterion_PD_singleMask(fake_B, abnm_mask, self.netD_abnm, target_is_real=True,
                                                          for_discriminator=False)
            loss_G_RGAN += loss_G_PG_abnm

        if torch.sum(nm_mask) > 0:
            loss_G_PG_nm = self.criterion_PD_singleMask(fake_B, nm_mask, self.netD_nm, target_is_real=True,
                                                        for_discriminator=False)
            loss_G_RGAN += loss_G_PG_nm

        loss_G = loss_G_RGAN + loss_G_HD
        writer.add_scalars('losses_G', {'loss_G': loss_G, 'loss_G_RGAN': loss_G_RGAN, 'loss_G_HD': loss_G_HD},
                           global_step=global_iter)
        return loss_G

    def criterion_GS(self, fake_B, real_B, real_A, nm_mask, abnm_mask, fake_C, real_C):
        loss_G = self.criterion_G(fake_B, real_B, real_A, nm_mask, abnm_mask)
        loss_S = self.criterion_S(fake_C, real_C)
        loss_full = loss_G + loss_S

        writer.add_scalars('losses_GS', {'loss_full': loss_G + loss_S, 'loss_S': loss_S, 'loss_G': loss_G},
                           global_step=global_iter)
        return loss_full


if __name__ == "__main__":
    """ Hpyer parameters """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--experiment_name', type=str, default='local_[1]')
    # training option
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_iters', type=int, default=100000)
    parser.add_argument('--schedule_times', type=int, default=20000)
    parser.add_argument('--schedule_decay_times', type=int, default=10000)
    parser.add_argument('--lr_g', type=float, default=0.0001)
    parser.add_argument('--lr_d', type=float, default=0.0004)
    parser.add_argument('--eval_iters', type=int, default=1000)  # 1000
    parser.add_argument('--save_iters', type=int, default=10000)  # 10000
    # data option
    datasetting.get_data_setting(parser, device_id=2)
    parser.add_argument('--train_folds', nargs='+', type=int, default=[0, 2, 3, 4])
    parser.add_argument('--test_folds', nargs='+', type=int, default=[1])
    parser.add_argument('--result_dir', type=str, default='MTL_P2PHD_PD')
    parser.add_argument('--test_img_save_dir', type=str, default='test_imgs')
    opt = parser.parse_args()

    opt.result_dir = os.path.join(opt.result_root, opt.result_dir, opt.experiment_name)
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
        print_options(parser, opt)
        shutil.copyfile(os.path.abspath(__file__), os.path.join(opt.result_dir, os.path.basename(__file__)))
    else:
        print("result_dir exists: ", opt.result_dir)
        exit()

    """ device configuration """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    """ track of experiments """
    writer = SummaryWriter(os.path.join(opt.result_dir, 'runs'))
    print("track view:", os.path.join(opt.result_dir, 'runs'))

    """ datasets and dataloader """
    print(opt.train_folds, opt.test_folds)
    train_transform = A.Compose([
        # A.Resize(512,512),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomCrop(width=512, height=512),
    ])
    test_transform = A.Compose([
        # A.Resize(512,512),
    ])
    train_dataset = AugmentedDataset(opt.A_dir, opt.B_dir, opt.C_dir, opt.D_dir, opt.train_folds, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_dataset = AugmentedDataset(opt.A_dir, opt.B_dir, opt.C_dir, opt.D_dir, opt.test_folds, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    print('train_dataset len:', len(train_dataset), 'test_dataset len:', len(test_loader))

    """ instantiate network and loss function"""
    model = DEGAN(opt)
    print_network(model, opt)

    """ training part """
    pbar = tqdm.tqdm(total=opt.num_iters)
    global_iter = 0
    model = model.to(device)
    model.train()
    best_lpips = 10
    best_iter = 0
    while global_iter < opt.num_iters:
        for _, (real_A, real_B, real_C, real_D, _) in enumerate(BackgroundGenerator(train_loader)):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            real_C = real_C.to(device)
            real_D = real_D.to(device)
            abnm_mask = real_C
            nm_mask = torch.abs(real_D - real_C)

            fake_B, fake_C = model.netG(real_A)
            # update D
            if torch.sum(nm_mask) > 0:
                set_requires_grad(model.netD_nm, True)
                loss_PD_nm = model.criterion_PD(fake_B, real_B, nm_mask, model.netD_nm)
                model.optimizer_D_nm.zero_grad()
                loss_PD_nm.backward()
                model.optimizer_D_nm.step()

            if torch.sum(abnm_mask) > 0:
                set_requires_grad(model.netD_abnm, True)
                loss_PD_abnm = model.criterion_PD(fake_B, real_B, abnm_mask, model.netD_abnm)
                model.optimizer_D_abnm.zero_grad()
                loss_PD_abnm.backward()
                model.optimizer_D_abnm.step()

            # update G
            set_requires_grad(model.netD_nm, False)
            set_requires_grad(model.netD_abnm, False)
            loss_G = model.criterion_GS(fake_B, real_B, real_A, nm_mask, abnm_mask, fake_C, real_C)
            model.optimizer_G.zero_grad()
            loss_G.backward()
            model.optimizer_G.step()

            global_iter += 1
            pbar.update(1)
            if global_iter % model.schedule_iters == 0:
                model.scheduler_G.step()
                model.scheduler_D_nm.step()
                model.scheduler_D_abnm.step()

            if global_iter % opt.eval_iters == 0:
                # model eval
                test_gen_lpips, test_seg_f1, test_seg_acc = 0, 0, 0
                model.eval()
                with torch.no_grad():
                    for _, (real_A, real_B, real_C, _, _) in enumerate(BackgroundGenerator(test_loader)):
                        real_A = real_A.to(device)
                        real_B = real_B.to(device)
                        fake_B, fake_C = model.netG(real_A)
                        fake_C = torch.sigmoid(fake_C)

                        test_gen_lpips += model.loss_fn_vgg(fake_B, real_B)
                        pred_out_np = (fake_C > 0.5).long().to('cpu').numpy()
                        pred_label_np = real_C.long().to('cpu').numpy()
                        test_seg_f1 += f1_score(pred_label_np.flatten(), pred_out_np.flatten(), average='micro')
                        test_seg_acc += np.sum(pred_out_np == pred_label_np) / (real_A.shape[2] * real_A.shape[3])
                model.train()

                # write testing data
                test_gen_lpips /= len(test_dataset)
                test_seg_f1 /= len(test_dataset)
                test_seg_acc /= len(test_dataset)
                writer.add_scalars('test_mertic', {'lpips': test_gen_lpips, 'acc': test_seg_acc, 'f1': test_seg_f1},
                                   global_iter)
                if global_iter == opt.eval_iters:
                    writer.add_images('test_real_A', real_A, global_iter)
                    writer.add_images('test_real_B', real_B, global_iter)
                    writer.add_images('test_real_C', real_B, global_iter)
                writer.add_images('test_fake_B', fake_B, global_iter)
                writer.add_images('test_fake_C', fake_C, global_iter)
                test_metrics = 'lp_%.3f_acc_%.3f_f1_%.3f' % (test_gen_lpips, test_seg_acc, test_seg_f1)

                # model saving
                save_dir = opt.result_dir
                if not os.path.exists(save_dir): os.mkdir(save_dir)
                if global_iter % opt.save_iters == 0:
                    state = {'netG': model.netG.state_dict()}
                    torch.save(state,
                               os.path.join(opt.result_dir, 'model_G_' + str(global_iter) + test_metrics + '.ckpt'))
                if test_gen_lpips < best_lpips:
                    last_path = os.path.join(opt.result_dir, 'model_G_best_'+str(best_iter)+'_'+'%.3f'%best_lpips+'.ckpt')
                    if os.path.exists(last_path): os.remove(last_path)
                    best_lpips = test_gen_lpips
                    best_iter = global_iter
                    state = {'netG': model.netG.state_dict()}
                    torch.save(state,
                               os.path.join(opt.result_dir, 'model_G_best_'+str(best_iter)+'_'+'%.3f'%best_lpips+'.ckpt'))
                pbar.set_postfix_str(s='test: ' + test_metrics+' best: %.3f iter:%d'%(best_lpips, best_iter))
            if global_iter == opt.num_iters: break

    opt.test_img_save_dir = os.path.join(opt.result_dir, opt.test_img_save_dir)
    if not os.path.exists(opt.test_img_save_dir): os.mkdir(opt.test_img_save_dir)
    imgdir = os.path.join(opt.test_img_save_dir, 'seg_bin')
    if not os.path.exists(imgdir): os.mkdir(imgdir)
    imgdir = os.path.join(opt.test_img_save_dir, 'seg_prb')
    if not os.path.exists(imgdir): os.mkdir(imgdir)
    imgdir = os.path.join(opt.test_img_save_dir, 'gen')
    if not os.path.exists(imgdir): os.mkdir(imgdir)
    # model_save_path = os.path.join(opt.result_dir,'model_GD_100000lp_0.332.ckpt')
    # model.netG.load_state_dict(torch.load(model_save_path)['netG'])
    model.netG.to(device)
    model.eval()
    ssim = 0
    with torch.no_grad():
        for i, (real_A, real_B, _, _, imgname) in enumerate(test_loader):
            real_A = real_A.to(device)
            fake_B, fake_C = model.netG(real_A)

            seg_prb_out = torch.sigmoid(fake_C)
            seg_bin_out = (seg_prb_out > 0.5).float()

            img_path = os.path.join(opt.test_img_save_dir, 'gen', imgname[0][:-4] + '.png')
            save_image(fake_B, img_path)

            img_path = os.path.join(opt.test_img_save_dir, 'seg_prb', imgname[0][:-4] + '.png')
            save_image(seg_prb_out, img_path)

            img_path = os.path.join(opt.test_img_save_dir, 'seg_bin', imgname[0][:-4] + '.png')
            save_image(seg_bin_out, img_path)
