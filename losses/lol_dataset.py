import cv2
import os
import torch
from torch.utils import data as data
import glob
import numpy as np

from basicsr.data.degradations import add_jpg_compression
from basicsr.data.transforms import augment, mod_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from scripts.utils import pad_tensor, hiseq_color_cv2_img, generate_position_encoding

@DATASET_REGISTRY.register()
class LoLDataset(data.Dataset):
    """Dataset for low-light image enhancement."""

    def __init__(self, opt):
        super(LoLDataset, self).__init__()
        self.opt = opt
        self.gt_root = opt['gt_root']
        self.input_root = opt['input_root']

        self.gt_paths = glob.glob(os.path.join(self.gt_root, '*.png')) + glob.glob(os.path.join(self.gt_root, '*.jpg'))

        self.mean = self.opt['mean']
        self.std = self.opt['std']

    # def __getitem__(self, index):
    #     gt_path = self.gt_paths[index]
    #     gt_name = os.path.split(gt_path)[-1]
    #     input_path = os.path.join(self.input_root, gt_name)
    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        gt_name = os.path.split(gt_path)[-1]
        
        # --- 添加下面这一行，把文件名里的 'normal' 替换成 'low' ---
        input_name = gt_name.replace('normal', 'low')
        
        # --- 使用修改后的 input_name 拼接路径 ---
        input_path = os.path.join(self.input_root, input_name)

        try:
            # 读取并转换颜色空间，同时进行归一化
            input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB) / 255.
            gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB) / 255.
        except Exception as e:
            print(f"Error loading images: {e}")
            raise ValueError(f"Failed to load images from {input_path} or {gt_path}")

        # 检查图像是否为空
        if input_img is None or gt_img is None:
            raise ValueError(f"Failed to load input or ground truth image from {input_path} or {gt_path}")

        # 执行 flip 操作 (在 concat 之前执行，保证图像是 3 通道)
        if self.opt.get('use_flip', False) and np.random.uniform() < 0.5:
            # print("Flipping images")  # 调试：确认 flip 操作
            input_img = cv2.flip(input_img, 1)
            gt_img = cv2.flip(gt_img, 1)

        # Mixup 处理
        if self.opt.get('LL_mixup_aug', False):
            if np.random.uniform() < 0.4:
                LL_mixup_aug_range = self.opt.get('LL_mixup_aug_range', [0.5, 1.])
                input_img = input_img * np.random.uniform(*LL_mixup_aug_range) + gt_img * (1 - np.random.uniform(*LL_mixup_aug_range))

        if self.opt.get('bright_aug', False):
            bright_aug_range = self.opt.get('bright_aug_range', [0.5, 1.5])
            input_img = input_img * np.random.uniform(*bright_aug_range)

        if self.opt.get('concat_with_hiseq', False):
            hiseql = cv2.cvtColor(hiseq_color_cv2_img(cv2.imread(input_path)), cv2.COLOR_BGR2RGB) / 255.
            if self.opt.get('hiseq_random_cat', False) and np.random.uniform(0, 1) < self.opt.get('hiseq_random_cat_p', 0.5):
                input_img = np.concatenate([hiseql, input_img], axis=2)
            else:
                input_img = np.concatenate([input_img, hiseql], axis=2)
            if self.opt.get('random_drop', False):
                if np.random.uniform() <= self.opt.get('random_drop_p', 1.0):
                    random_drop_val = self.opt.get('random_drop_val', 0)
                    if np.random.uniform() < 0.5:
                        input_img[:, :, :3] = random_drop_val
                    else:
                        input_img[:, :, 3:] = random_drop_val
            if self.opt.get('random_drop_hiseq', False):
                if np.random.uniform() < 0.5:
                    input_img[:, :, 3:] = 0

        if self.opt['input_mode'] == 'crop':
            crop_size = self.opt['crop_size']
            H, W, _ = input_img.shape
            assert input_img.shape[:2] == gt_img.shape[:2], f"{input_img.shape}, {gt_img.shape}, {gt_path}"
            h = np.random.randint(0, H - crop_size + 1)
            w = np.random.randint(0, W - crop_size + 1)
            gt_img = gt_img[h: h + crop_size, w: w + crop_size, :]
            input_img = input_img[h: h + crop_size, w: w + crop_size, :]

        if self.opt['input_mode'] == 'pad':
            divide = self.opt['divide']
            # 使用 np.ascontiguousarray 保证内存连续性
            gt_img_pt = torch.from_numpy(np.ascontiguousarray(gt_img.transpose((2, 0, 1))))
            input_img_pt = torch.from_numpy(np.ascontiguousarray(input_img.transpose((2, 0, 1))))
            gt_img_pt = torch.unsqueeze(gt_img_pt, 0)
            input_img_pt = torch.unsqueeze(input_img_pt, 0)
            gt_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gt_img_pt, divide)
            input_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input_img_pt, divide)
            gt_img_pt = gt_img_pt[0, ...]
            input_img_pt = input_img_pt[0, ...]
            gt_img = gt_img_pt.numpy().transpose((1, 2, 0))
            input_img = input_img_pt.numpy().transpose((1, 2, 0))

        # Ensure that the images are in the right format
        # 同样使用 np.ascontiguousarray 保证内存连续性
        gt_img_pt = torch.from_numpy(np.ascontiguousarray(gt_img.transpose((2, 0, 1)))).float()
        input_img_pt = torch.from_numpy(np.ascontiguousarray(input_img.transpose((2, 0, 1)))).float()

        return_dict = {'lq': input_img_pt, 'gt': gt_img_pt, 'lq_path': input_path, 'gt_path': gt_path}
        if self.opt['input_mode'] == 'pad':
            return_dict["pad_left"] = pad_left
            return_dict["pad_right"] = pad_right
            return_dict["pad_top"] = pad_top
            return_dict["pad_bottom"] = pad_bottom

        return return_dict

    def __len__(self):
        return len(self.gt_paths)