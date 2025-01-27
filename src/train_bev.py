"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .bevdataset import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info


def train(  dataroot='/home/mbarin/storage/data-transfuser/',
            nepochs= 400, #10000,
            gpuid=1,

            H=160, W=320, #H=900, W=1600,
            resize_lim=(0.193, 0.225),
            final_dim=(160, 320), #(128, 352),
            bot_pct_lim=(0.0, 0.0), #0.22),
            rot_lim=(0,0),  #(-5.4, 5.4),
            rand_flip= False, #True,
            ncams=3,
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',

            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=1, #4,
            nworkers=10,
            lr=1e-3,
            weight_decay=1e-7,
            ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             #'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                             ],
                    'Ncams': ncams,
                }
    trainloader, valloader = compile_data(dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          #parser_name='segmentationdata'
                                          )

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 10 #1000 if version == 'mini' else 

    model.train()
    counter = 0
    """ (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) = next(iter(trainloader))

    import matplotlib.pyplot as plt
    plt.imshow(   binimgs[0].permute(1, 2, 0).cpu().numpy()  )
    plt.savefig(f'logs/binimg_FULL_bev.png')

    print('BINNNG ', binimgs.shape)
    binimgs_cropped = binimgs[0,:,100:,:]
    binimgs = binimgs_cropped """

    for epoch in range(nepochs):
        print('#### EPOCH ', epoch)
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            """ 
            print('PREDS ', type(preds), preds.shape)
            print('BINIMGS ', type(binimgs) , binimgs.shape) 
            """

            import matplotlib.pyplot as plt
            plt.imshow(   binimgs[0].permute(1, 2, 0).cpu().numpy()  )
            plt.savefig(f'logs/binimg_FULL_bev.png')

            #print('BINNNG ', binimgs.shape)
            binimgs_cropped = binimgs[0,:,100:,:]
            binimgs = binimgs_cropped

            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 1 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)

            if counter % 1 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                valloader = [next(iter(valloader))] #[(imgs, rots, trans, intrins, post_rots, post_trans, binimgs)] # TODO: Added for overfitting
                val_info = get_val_info(model, epoch, valloader, loss_fn, device)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()
