import os
import cv2 
import perspective_and_equirectangular.lib.Equirec2Perspec as E2P
import perspective_and_equirectangular.lib.Perspec2Equirec as P2E
import perspective_and_equirectangular.lib.multi_Perspec2Equirec as m_P2E
import glob
import argparse


import numpy as np
from scipy.ndimage import map_coordinates
import cv2
import torch.nn as nn
import torch

import os
import sys
import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Cube2Equirec(nn.Module):
    def __init__(self, face_w, equ_h, equ_w):
        super(Cube2Equirec, self).__init__()
        '''
        face_w: int, the length of each face of the cubemap
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        '''

        self.face_w = face_w
        self.equ_h = equ_h
        self.equ_w = equ_w


        # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
        self._equirect_facetype()
        self._equirect_faceuv()


    def _equirect_facetype(self):
        '''
        0F 1R 2B 3L 4U 5D
        '''
        tp = np.roll(np.arange(4).repeat(self.equ_w // 4)[None, :].repeat(self.equ_h, 0), 3 * self.equ_w // 8, 1)

        # Prepare ceil mask
        mask = np.zeros((self.equ_h, self.equ_w // 4), np.bool)
        idx = np.linspace(-np.pi, np.pi, self.equ_w // 4) / 4
        idx = self.equ_h // 2 - np.round(np.arctan(np.cos(idx)) * self.equ_h / np.pi).astype(int)
        for i, j in enumerate(idx):
            mask[:j, i] = 1
        mask = np.roll(np.concatenate([mask] * 4, 1), 3 * self.equ_w // 8, 1)

        tp[mask] = 4
        tp[np.flip(mask, 0)] = 5

        self.tp = tp
        self.mask = mask

    def _equirect_faceuv(self):

        lon = ((np.linspace(0, self.equ_w -1, num=self.equ_w, dtype=np.float32 ) +0.5 ) /self.equ_w - 0.5 ) * 2 *np.pi
        lat = -((np.linspace(0, self.equ_h -1, num=self.equ_h, dtype=np.float32 ) +0.5 ) /self.equ_h -0.5) * np.pi

        lon, lat = np.meshgrid(lon, lat)

        coor_u = np.zeros((self.equ_h, self.equ_w), dtype=np.float32)
        coor_v = np.zeros((self.equ_h, self.equ_w), dtype=np.float32)

        for i in range(4):
            mask = (self.tp == i)
            coor_u[mask] = 0.5 * np.tan(lon[mask] - np.pi * i / 2)
            coor_v[mask] = -0.5 * np.tan(lat[mask]) / np.cos(lon[mask] - np.pi * i / 2)

        mask = (self.tp == 4)
        c = 0.5 * np.tan(np.pi / 2 - lat[mask])
        coor_u[mask] = c * np.sin(lon[mask])
        coor_v[mask] = c * np.cos(lon[mask])

        mask = (self.tp == 5)
        c = 0.5 * np.tan(np.pi / 2 - np.abs(lat[mask]))
        coor_u[mask] = c * np.sin(lon[mask])
        coor_v[mask] = -c * np.cos(lon[mask])

        # Final renormalize
        coor_u = (np.clip(coor_u, -0.5, 0.5)) * 2
        coor_v = (np.clip(coor_v, -0.5, 0.5)) * 2

        # Convert to torch tensor
        self.tp = torch.from_numpy(self.tp.astype(np.float32) / 2.5 - 1)
        self.coor_u = torch.from_numpy(coor_u)
        self.coor_v = torch.from_numpy(coor_v)

        sample_grid = torch.stack([self.coor_u, self.coor_v, self.tp], dim=-1).view(1, 1, self.equ_h, self.equ_w, 3)
        self.sample_grid = nn.Parameter(sample_grid, requires_grad=False)

    def forward(self, cube_feat):

        bs, ch, h, w = cube_feat.shape
        assert h == self.face_w and w // 6 == self.face_w

        cube_feat = cube_feat.view(bs, ch, 1,  h, w)
        cube_feat = torch.cat(torch.split(cube_feat, self.face_w, dim=-1), dim=2)

        cube_feat = cube_feat.view([bs, ch, 6, self.face_w, self.face_w])
        sample_grid = torch.cat(bs * [self.sample_grid], dim=0)
        equi_feat = F.grid_sample(cube_feat, sample_grid, padding_mode="border", align_corners=True)

        return equi_feat.squeeze(2)

# Based on https://github.com/sunset1995/py360convert
class Equirec2Cube:
    def __init__(self, equ_h, equ_w, face_w, padding):
        '''
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        face_w: int, the length of each face of the cubemap
        '''

        self.equ_h = equ_h
        self.equ_w = equ_w
        self.pad = padding
        self.oface_w = face_w
        if self.pad:
          self.face_w = face_w+2*padding
        else:
          self.face_w = face_w
        self._xyzcube()
        self._xyz2coor()

        # For convert R-distance to Z-depth for CubeMaps
        cosmap = 1 / np.sqrt((2 * self.grid[..., 0]) ** 2 + (2 * self.grid[..., 1]) ** 2 + 1)
        self.cosmaps = np.concatenate(6 * [cosmap], axis=1)[..., np.newaxis]

    def _xyzcube(self):
        '''
        Compute the xyz cordinates of the unit cube in [F R B L U D] format.
        '''
        self.xyz = np.zeros((self.face_w, self.face_w * 6, 3), np.float32)
        if self.pad:
          rng = np.linspace(-0.5 - self.pad / self.oface_w, 0.5 + self.pad / self.oface_w, num=self.face_w)
        else:
          rng = np.linspace(-0.5, 0.5, num=self.face_w, dtype=np.float32)
        self.grid = np.stack(np.meshgrid(rng, -rng), -1)

        # Front face (z = 0.5)
        self.xyz[:, 0 * self.face_w:1 * self.face_w, [0, 1]] = self.grid
        self.xyz[:, 0 * self.face_w:1 * self.face_w, 2] = 0.5

        # Right face (x = 0.5)
        self.xyz[:, 1 * self.face_w:2 * self.face_w, [2, 1]] = self.grid[:, ::-1]
        self.xyz[:, 1 * self.face_w:2 * self.face_w, 0] = 0.5

        # Back face (z = -0.5)
        self.xyz[:, 2 * self.face_w:3 * self.face_w, [0, 1]] = self.grid[:, ::-1]
        self.xyz[:, 2 * self.face_w:3 * self.face_w, 2] = -0.5

        # Left face (x = -0.5)
        self.xyz[:, 3 * self.face_w:4 * self.face_w, [2, 1]] = self.grid
        self.xyz[:, 3 * self.face_w:4 * self.face_w, 0] = -0.5

        # Up face (y = 0.5)
        self.xyz[:, 4 * self.face_w:5 * self.face_w, [0, 2]] = self.grid[::-1, :]
        self.xyz[:, 4 * self.face_w:5 * self.face_w, 1] = 0.5

        # Down face (y = -0.5)
        self.xyz[:, 5 * self.face_w:6 * self.face_w, [0, 2]] = self.grid
        self.xyz[:, 5 * self.face_w:6 * self.face_w, 1] = -0.5

    def _xyz2coor(self):

        # x, y, z to longitude and latitude
        x, y, z = np.split(self.xyz, 3, axis=-1)
        lon = np.arctan2(x, z)
        c = np.sqrt(x ** 2 + z ** 2)
        lat = np.arctan2(y, c)

        # longitude and latitude to equirectangular coordinate
        self.coor_x = (lon / (2 * np.pi) + 0.5) * self.equ_w - 0.5
        self.coor_y = (-lat / np.pi + 0.5) * self.equ_h - 0.5

    def sample_equirec(self, e_img, order=0):
        pad_u = np.roll(e_img[[0]], self.equ_w // 2, 1)
        pad_d = np.roll(e_img[[-1]], self.equ_w // 2, 1)
        e_img = np.concatenate([e_img, pad_d, pad_u], 0)
        # pad_l = e_img[:, [0]]
        # pad_r = e_img[:, [-1]]
        # e_img = np.concatenate([e_img, pad_l, pad_r], 1)
        
        #print(self.coor_y.shape)
        #print(self.coor_x.shape)

        return map_coordinates(e_img, [self.coor_y, self.coor_x],
                               order=order, mode='wrap')[..., 0]

    def run(self, equ_img, equ_dep=None):

        h, w = equ_img.shape[:2]
        if h != self.equ_h or w != self.equ_w:
            equ_img = cv2.resize(equ_img, (self.equ_w, self.equ_h))
            if equ_dep is not None:
                equ_dep = cv2.resize(equ_dep, (self.equ_w, self.equ_h), interpolation=cv2.INTER_NEAREST)

        cube_img = np.stack([self.sample_equirec(equ_img[..., i], order=1)
                             for i in range(equ_img.shape[2])], axis=-1)

        if equ_dep is not None:
            cube_dep = np.stack([self.sample_equirec(equ_dep[..., i], order=0)
                                 for i in range(equ_dep.shape[2])], axis=-1)
            cube_dep = cube_dep * self.cosmaps

        if equ_dep is not None:
            return cube_img, cube_dep
        else:
            return cube_img

def panorama2cube4(input_dir):
    base_dir = os.path.basename(input_dir.rstrip('/\\'))
    if not base_dir:
        base_dir = 'images'

    output_dir = os.path.join(input_dir[:-6], base_dir + '_split/')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_image = sorted(glob.glob(input_dir + '/*.*'))
    height, width = cv2.imread(all_image[0]).shape[:2]
    paddingsize = 1
    e2c = Equirec2Cube(height, width, height // 2, paddingsize)
    cube_size = int(width / 4)+2*paddingsize

    for index in range(len(all_image)):
        #equ = E2P.Equirectangular(all_image[index])    # Load equirectangular image
        
        ERP = cv2.imread(all_image[index], cv2.IMREAD_COLOR)
        CUBE = e2c.run(ERP)
        #此部分验证sp的有效性
        '''
        cube=[]
        for j in range(6):
          tmp = CUBE[:, cube_size*j:cube_size*(j+1),:]
          tmp = torch.from_numpy(tmp).unsqueeze(0).cuda()
          tmp = tmp.permute(0, 3, 1, 2).float()
          tmp = tmp[:,:, 1:-1, 1:-1]
          cube.append(tmp)        
        c2e = Cube2Equirec(480, 960, 1920).cuda()
        cubet = torch.cat(cube, dim=-1)
        print(cubet.shape)
        erp = c2e(cubet).squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        error = ERP-erp
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'eerror.jpg'        
        cv2.imwrite(output1, error) 
        '''                   
                                    
        #cubet = torch.cat(sortedcube, dim=0)
        img = CUBE[:, :cube_size,:]#CUBE[:, :cube_size,:]  # Specify parameters(FOV, theta, phi, height, width)    
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'F0.jpg'        
        cv2.imwrite(output1, img)
        img = CUBE[:, cube_size:cube_size*2,:]#CUBE[:, :cube_size,:]  # Specify parameters(FOV, theta, phi, height, width)    
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'R0.jpg'        
        cv2.imwrite(output1, img)
        img = CUBE[:, cube_size*2:cube_size*3,:]#CUBE[:, :cube_size,:]  # Specify parameters(FOV, theta, phi, height, width)    
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'B0.jpg'        
        cv2.imwrite(output1, img)
        img = CUBE[:, cube_size*3:cube_size*4,:]#CUBE[:, :cube_size,:]  # Specify parameters(FOV, theta, phi, height, width)    
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'L0.jpg'        
        cv2.imwrite(output1, img)
        img = CUBE[:,cube_size*4:cube_size*5,:]#CUBE[:, :cube_size,:]  # Specify parameters(FOV, theta, phi, height, width)    
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'U0.jpg'        
        cv2.imwrite(output1, img)
        img = CUBE[:,cube_size*5:cube_size*6,:]#CUBE[:, :cube_size,:]  # Specify parameters(FOV, theta, phi, height, width)    
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'D0.jpg'        
        cv2.imwrite(output1, img)
        
        h, w, c = ERP.shape
        angle_radians = np.radians(45)
        shift = int(angle_radians / (2 * np.pi) * w)
        rotated_image = np.roll(ERP, shift=shift, axis=1)  # 水平平移
        CUBE = e2c.run(rotated_image)
        
        img = CUBE[:, :cube_size,:]#CUBE[:, :cube_size,:]  # Specify parameters(FOV, theta, phi, height, width)    
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'F1.jpg'        
        cv2.imwrite(output1, img)
        img = CUBE[:, cube_size:cube_size*2,:]#CUBE[:, :cube_size,:]  # Specify parameters(FOV, theta, phi, height, width)    
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'R1.jpg'        
        cv2.imwrite(output1, img)
        img = CUBE[:, cube_size*2:cube_size*3,:]#CUBE[:, :cube_size,:]  # Specify parameters(FOV, theta, phi, height, width)    
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'B1.jpg'        
        cv2.imwrite(output1, img)
        img = CUBE[:, cube_size*3:cube_size*4,:]#CUBE[:, :cube_size,:]  # Specify parameters(FOV, theta, phi, height, width)    
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'L1.jpg'        
        cv2.imwrite(output1, img)
        img = CUBE[:,cube_size*4:cube_size*5,:]#CUBE[:, :cube_size,:]  # Specify parameters(FOV, theta, phi, height, width)    
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'U1.jpg'        
        cv2.imwrite(output1, img)
        img = CUBE[:,cube_size*5:cube_size*6,:]#CUBE[:, :cube_size,:]  # Specify parameters(FOV, theta, phi, height, width)    
        output1 = output_dir + os.path.splitext(os.path.basename(all_image[index]))[0] + 'D1.jpg'        
        cv2.imwrite(output1, img)

def panorama2cube(input_dir):
    base_dir = os.path.basename(input_dir.rstrip('/\\'))
    if not base_dir:
        base_dir = 'images'

    all_image = sorted(glob.glob(input_dir + '/*.*'))
    height, width = cv2.imread(all_image[0]).shape[:2]
    cube_size = int(width / 4)

    for index in range(len(all_image)):
        equ = E2P.Equirectangular(all_image[index])    # Load equirectangular image

        out_img = input_dir + '/' + os.path.basename(all_image[index])
        img_0 = equ.GetPerspective(90, 0, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)python cube_convert.py datasets/Ricoh360/center/images --split
        img_right = equ.GetPerspective(90, 90, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        img_left = equ.GetPerspective(90, -90, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)
        img_back = equ.GetPerspective(90, 180, 0, cube_size, cube_size)  # Specify parameters(FOV, theta, phi, height, width)

        img = cv2.hconcat([img_left, img_0, img_right, img_back])
        cv2.imwrite(out_img, img)

def main():
    parser = argparse.ArgumentParser(description="Convert equirectangular panorama to cube map.")
    parser.add_argument("input_dir", type=str, help="Input directory containing equirectangular images.")
    parser.add_argument("--split", action='store_true', help="Split the panorama into 4 images (front, right, back, left)")

    args = parser.parse_args()

    if args.split:
        panorama2cube4(args.input_dir)
    else:
        panorama2cube(args.input_dir)

if __name__ == "__main__":
    main()
