"""
code adapted from https://github.com/nv-tlabs/lift-splat-shoot
and also https://github.com/wayveai/fiery/blob/master/fiery/data.py
"""

import torch
import os
import numpy as np
from PIL import Image
from glob import glob

import time

import matplotlib.pyplot as plt

from .tools import img_transform, normalize_img, gen_dx_bx



def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0


def get_val_info(model, valloader, loss_fn, device, use_tqdm=False, max_iters=None, use_lidar=False):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print('running eval...')

    t0 = time()
    
    loader = tqdm(valloader) if use_tqdm else valloader

    if max_iters is not None:
        counter = 0
    with torch.no_grad():
        for batch in loader:

            if max_iters is not None:
                counter += 1
                if counter > max_iters:
                    break

            if use_lidar:
                allimgs, rots, trans, intrins, pts, binimgs = batch
            else:
                allimgs, rots, trans, intrins, binimgs = batch
                
            preds = model(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device))
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds[:,0:1], binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds[:,0:1], binimgs)
            total_intersect += intersect
            total_union += union
    t1 = time()
    print('eval took %.2f seconds' % (t1-t0))

    model.train()

    if max_iters is not None:
        normalizer = counter
    else:
        normalizer = len(valloader.dataset)
        
    return {
        'total_loss': total_loss / normalizer,
        'iou': total_intersect / total_union,
    }



import os 
import glob
import json

import numpy as np
import torch

from PIL import Image
from torchvision import transforms
#from torchvision.datasets.folder import default_loader

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from typing import List, Optional, Sequence, Union, Callable

import matplotlib.pyplot as plt

from .bev_utils import * 
from .tools import gen_dx_bx


########## SENSOR CONFIGS FOR TRANSFUSER DATA #############

cam_config = {
            'width': 320,
            'height': 160,
            'fov': 60
        }

SENSOR_CONFIGS = {
        'CAM_RGB_FRONT_LEFT': {
                            'type': 'sensor.camera.rgb',
                            'x': 1.3, 'y': 0.0, 'z':2.3,
                            'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                            'width': cam_config['width'], 'height': cam_config['height'], 'fov': cam_config['fov'],
                            'id': 'rgb_left'
        },
        'CAM_RGB_FRONT': {  
                            'type': 'sensor.camera.rgb',
                            'x': 1.3, 'y': 0.0, 'z':2.3,
                            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                            'width': cam_config['width'], 'height': cam_config['height'], 'fov': cam_config['fov'],
                            'id': 'rgb_front'
                            },

        'CAM_RGB_FRONT_RIGHT': {
                            'type': 'sensor.camera.rgb',
                            'x': 1.3, 'y': 0.0, 'z':2.3,
                            'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                            'width': cam_config['width'], 'height': cam_config['height'], 'fov': cam_config['fov'],
                            'id': 'rgb_right'
        }
}

class BEVDataset(Dataset):
    def __init__(self, 
                data_path,
                split, 
                image_folder,
                transform,
                zero_out_red_channel,
                data_aug_conf,
                rgb_cam_configs = {},
                use_radar_filters = False,
                do_shuffle_cams = False,
            ):

        super(BEVDataset, self).__init__()


        self.data_path = data_path #'/home/mbarin/storage/data-transfuser'
        self.split = split
        self.image_folder = image_folder
        self.transform = transform
        self.zero_out_red_channel = zero_out_red_channel
        self.data_aug_conf = data_aug_conf
        
        self.dataset = self.get_img_paths()

        #self.cams = cams
        self.rgb_cam_configs = SENSOR_CONFIGS # rgb_cam_configs
        self.use_radar_filters = use_radar_filters
        self.do_shuffle_cams = do_shuffle_cams
        # self.res_3d = res_3d
        # self.bounds = bounds
        #self.centroid = centroid

        self.seqlen = 1
        self.refcam_id = 1 # TODO: check if it is the front camera


        #XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = self.bounds
        self.Z, self.Y, self.X = 200, 0 , 200 #self.res_3d


        grid_conf = { # note the downstream util uses a different XYZ ordering
            'xbound': [-50.0, 50.0, 0.5], #[XMIN, XMAX, (XMAX-XMIN)/float(self.X)],
            'ybound': [-50.0, 50.0, 0.5], #[ZMIN, ZMAX, (ZMAX-ZMIN)/float(self.Z)],
            'zbound': [-10.0, 10.0, 20.0] #[YMIN, YMAX, (YMAX-YMIN)/float(self.Y)],
        }

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()



    def get_img_paths(self):

        self.main_folders = os.listdir(self.data_path)

        if self.split=="train":
            self.main_folders = [self.main_folders[2]]  # [:-2] # TODO: is it for rgb [2:-2]
        elif self.split=="val":
            self.main_folders = [self.main_folders[2]]   #[self.main_folders[-2]]
        elif self.split=="test":
            self.main_folders = [self.main_folders[-2]]

        #print('main folders ', self.main_folders)


        sub_folder_depth = '/*/*'

        self.sub_folders = []
        for folder in self.main_folders:
            sub_folders_path = self.data_path + folder + sub_folder_depth
            self.sub_folders += glob.glob(sub_folders_path)

        self.images = []
        dataset = []
        for folder in self.sub_folders:
            img_paths = folder + '/' + self.image_folder + '/*'

            img_paths = sorted(glob.glob(img_paths))

            # TODO: try passing multiple steps
            dataset += [path for path in img_paths]


        #dataset = [dataset[49]] # YOU TRY OVERFITTING!!!!

        #print('DATASET ' , dataset)


        print(f"Detected {len(dataset)} images in split {self.split}")
        print(f"Detected {len(dataset)} commands in split {self.split}")

        return dataset
    
    
    def get_file_path(self, rgb_path, folder_name, file_ext, is_encoded=False):

        if is_encoded:
            file_name = 'encoded_' + rgb_path.split('/')[-1][:-3]  + file_ext
        else:
            file_name = rgb_path.split('/')[-1][:-3] + file_ext

        # print('FOLDER NAME ', folder_name)
        # print('FILE NAME ', file_name)
        # print('PATH LIST ', rgb_path.split('/')[:-2])
        file_path = '/'.join(rgb_path.split('/')[:-2] + [folder_name] + [file_name])

        #print('fILE PATH ', file_path)

        return file_path
        

    def __len__(self):

        return len(self.dataset)


    def get_egopose_4x4matrix(self, rgb_path):

        f =  open(self.get_file_path(rgb_path,'measurements','json'))
        measurements = json.load(f)
        egopose = measurements['ego_matrix']

        return torch.Tensor(egopose)



    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        #print('H ', H, ' W ', W) 
        fH, fW = self.data_aug_conf['final_dim']

        #print('ffH ', fH, 'fW ', fW )
        if self.split == 'train':
            #resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize = 1
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate


    def get_binimg(self, rgb_path):
        '''
            get_binimg():
            For NuScenes case, they define polygons representing vehicles based on the egopose.
            Then form the topdown view of the scene.

            get_seg_bev():
            Also, they form bev segmentation of the scene using LIDAR data. 
            We consider topdown view as ground truth BEV. 

            For Transfuser data, we have 'topdown' and 'semantics' folders. The former corresponds to their binimg.
            The latter is the segmentic version of the rgb camera images. 960 x 160     

            For ncams = 3, we need to crop topdown images and get the upper half part of the image. 
            Since we are just predicting the front part.
        '''

        topdown_path = self.get_file_path(rgb_path,'topdown','png',is_encoded=True)
        topdown = Image.open(topdown_path)
        topdown = transforms.ToTensor()(topdown) # .unsqueeze_(0)
        
        # zero out red & blue channel
        topdown[0,:,:] = 0
        topdown[2,:,:] = 0

        # take the upper part of the topdown view 
        _, H, W = topdown.shape
        topdown_t = topdown #topdown[:,:H//2,:] #[:,H//2:(H//2+self.Z),:]
        

        #print('topdowb shape ' , topdown_t.shape)

        """ plt.imshow(topdown_t.permute(1,2,0))
        plt.show() """

        # TODO: transform topdown img based on model input img. 
        topdown =  transforms.functional.center_crop(topdown_t,(self.X,self.Z))
        topdown = transforms.functional.rotate(topdown,180)
        topdown = transforms.functional.rgb_to_grayscale(topdown).squeeze()

        topdown[topdown!=0.0]=1

        return topdown.unsqueeze(0)

    
    def get_image_data(self, rgb_path):

        rgbs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        imgs = []

        # RGB CAMERA IMAGE
        rgb = Image.open(rgb_path)

        convert_tensor = transforms.ToTensor()
        rgb = convert_tensor(rgb)
        
        if self.zero_out_red_channel:
            rgb[0, :, :] = 0

        _, H, W = rgb.shape

        rgbs.append(rgb[:,:,:W//3])
        rgbs.append(rgb[:,:,W//3:2*W//3])
        rgbs.append(rgb[:,:,2*W//3:])

        import json
        #measurements = json.load(self.get_measurements_path(rgb_path))
        _, trans, rots, intrins = get_trans_and_rot_from_sensor_list(SENSOR_CONFIGS) #bev_utils.
  

        """ print('RGBs ', len(rgbs))
        print(rgbs[0].shape)
        print('TRNS ' , len(trans))
        print('TRNS ' , trans[0].shape)
        print('TOR ' , rots[0].shape)
        print('INSTRINS ', len(intrins))
        print('INTRINSS ', intrins[0].shape) """

        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        for idx, img in enumerate(rgbs):
            
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            """ print('INDEX ' , idx)
            print(torch.max(img), torch.min(img)) """

            plt.imshow( img.permute(1, 2, 0).cpu().numpy()  )
            plt.savefig(f'./images/rgb{idx}.png')

            """ print('POST ROT ', post_rot)
            print('POST TRAN ', post_tran) """

            img = to_pil(img)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            
            plt.imshow( to_tensor(img).permute(1, 2, 0).cpu().numpy()  )
            plt.savefig(f'./images/after_transform_rgb{idx}.png')

            imgs.append(normalize_img(img))
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return [torch.stack(rgbs), torch.stack(rots), torch.stack(trans), torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans)]


    def get_single_item(self,rgb_path):


        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rgb_path)  # rot , trans of the rgb cameras
        
        egopose = self.get_egopose_4x4matrix(rgb_path)
        binimg = self.get_binimg(rgb_path)

        """ print('SEG BEV SHAPE ', seg_bev.shape)
        print('seg bev ', torch.max(seg_bev), torch.max(valid_bev))
        print('seg bev ', torch.min(seg_bev), torch.min(valid_bev))
        import matplotlib.pyplot as plt
        plt.imshow(seg_bev[0])
        plt.show()

        plt.imshow(valid_bev[0])
        plt.show()
        """

        #print('#### CAMS ', cams)
        """ print('#### IMGS ', type(imgs), imgs.shape)
        print('#### GT ', type(binimg), binimg.shape)
        print('#### rots ', rots.shape, rots)
        print('#### trans ', trans.shape, trans)
        print('#### post_rots ', post_rots)
        print('#### post_ trans ', post_trans)

        exit()
        """
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


    def __getitem__(self, idx):

        rgb_path = self.dataset[idx]

        return self.get_single_item(rgb_path)

 



def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(data_dir, 
                 data_aug_conf,
                 grid_conf, 
                 bsz,
                 nworkers, 
                 shuffle=True, 
                 ):

    
    print('loading bev dataset...')
    #print('version ', version)
    print('dataroot ', data_dir)


    traindata = BEVDataset(data_path=data_dir,
                            split='train', 
                            image_folder='rgb',
                            transform = None,
                            zero_out_red_channel=True,
                            data_aug_conf=data_aug_conf,
                            )
    valdata = BEVDataset(data_path=data_dir,
                            split='val', 
                            image_folder='rgb',
                            transform = None,
                            zero_out_red_channel=True,
                            data_aug_conf=data_aug_conf,
                            )

    trainloader = torch.utils.data.DataLoader(
        traindata,
        batch_size=bsz,
        shuffle=shuffle,
        num_workers=nworkers,
        drop_last=True,
        worker_init_fn=worker_rnd_init,
        pin_memory=False)


    valloader = torch.utils.data.DataLoader(
        valdata,
        batch_size=bsz,
        shuffle=shuffle,
        num_workers=nworkers,
        drop_last=True,
        pin_memory=False)


    print('data ready')


    return trainloader, valloader
