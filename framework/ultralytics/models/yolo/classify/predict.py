# Ultralytics  AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
import torchvision.transforms as T


class ClassificationPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a classification model.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes ClassificationPredictor setting the task to 'classify'."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"
        self._legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"

        self.model_channels = self.args.channels

    def preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )

            if is_legacy_transform:  # to handle legacy transforms
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                # numpy,PIL
                processed_imgs = []
                for im in img:
                    # numpy
                    if not isinstance(im, np.ndarray):
                        im = np.array(im)

                    # 
                    current_channels = im.shape[2] if len(im.shape) == 3 else 1

                    # 
                    if self.model_channels == 1:
                        # 1:1
                        if len(im.shape) == 2:
                            im_processed = im  # 
                        elif current_channels == 1:
                            im_processed = im[:, :, 0]  # 
                        else:
                            # 
                            im_processed = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                    elif self.model_channels == 3:
                        # 3:3
                        if len(im.shape) == 2:
                            # 3
                            im_processed = np.stack([im] * 3, axis=2)
                        elif current_channels == 1:
                            # 3
                            im_processed = np.repeat(im, 3, axis=2)
                        elif current_channels == 3:
                            # 3
                            im_processed = im
                        elif current_channels == 4:
                            # 43
                            im_processed = im[:, :, :3]
                        elif current_channels == 6:
                            # 63
                            im_processed = im[:, :, :3]
                        else:
                            # 3
                            im_processed = im[:, :, :3]

                    elif self.model_channels == 4:
                        # 4:4
                        if len(im.shape) == 2:
                            # 4,
                            raise ValueError("1-channel image cannot be converted to 4 channels for 4-channel model")
                        elif current_channels == 1:
                            # 4,
                            raise ValueError("1-channel image cannot be converted to 4 channels for 4-channel model")
                        elif current_channels == 3:
                            # 34,
                            raise ValueError("3-channel image cannot be converted to 4 channels for 4-channel model")
                        elif current_channels == 4:
                            # 4
                            im_processed = im
                        elif current_channels == 6:
                            # 64
                            im_processed = im[:, :, :4]
                        else:
                            # 4
                            im_processed = im[:, :, :min(current_channels, 4)]
                            if current_channels < 4:
                                raise ValueError(
                                    f"{current_channels}-channel image cannot be converted to 4 channels for 4-channel model")

                    elif self.model_channels == 6:
                        # 6:6
                        if len(im.shape) == 2:
                            # 6,
                            raise ValueError("1-channel image cannot be converted to 6 channels for 6-channel model")
                        elif current_channels == 1:
                            # 6,
                            raise ValueError("1-channel image cannot be converted to 6 channels for 6-channel model")
                        elif current_channels == 3:
                            # 36,
                            raise ValueError("3-channel image cannot be converted to 6 channels for 6-channel model")
                        elif current_channels == 4:
                            # 46,
                            raise ValueError("4-channel image cannot be converted to 6 channels for 6-channel model")
                        elif current_channels == 6:
                            # 6
                            im_processed = im
                        else:
                            # 6
                            im_processed = im[:, :, :min(current_channels, 6)]
                            if current_channels < 6:
                                raise ValueError(
                                    f"{current_channels}-channel image cannot be converted to 6 channels for 6-channel model")

                    else:
                        # :
                        if current_channels != self.model_channels:
                            raise ValueError(
                                f"{current_channels}-channel image does not match {self.model_channels}-channel model")
                        im_processed = im

                    # 
                    im_processed = self._apply_geometric_transforms(im_processed)

                    # tensor[0,1]
                    tensor_img = torch.from_numpy(im_processed.transpose(2, 0, 1)).float() / 255.0

                    # 
                    tensor_img = self._apply_normalization(tensor_img)

                    processed_imgs.append(tensor_img)

                img = torch.stack(processed_imgs, dim=0)

        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()

    def _apply_geometric_transforms(self, img):
        """(resizecentercrop)"""
        # transform
        import torchvision.transforms as T

        # resize
        resize_size = 224  # 
        for transform in self.transforms.transforms:
            if isinstance(transform, T.Resize):
                if isinstance(transform.size, int):
                    resize_size = (transform.size, transform.size)
                else:
                    resize_size = transform.size
                break

        # resize
        h, w = img.shape[:2]
        if isinstance(resize_size, int):
            resize_size = (resize_size, resize_size)

        if (h, w) != resize_size:
            # resize
            if self.model_channels == 1:
                # 1:resize
                img = cv2.resize(img, resize_size, interpolation=cv2.INTER_LINEAR)
            elif self.model_channels == 3:
                # 3:resize
                img = cv2.resize(img, resize_size, interpolation=cv2.INTER_LINEAR)
            elif self.model_channels == 4:
                # 4:3resize,4resize
                rgb_resized = cv2.resize(img[:, :, :3], resize_size, interpolation=cv2.INTER_LINEAR)
                fourth_resized = cv2.resize(img[:, :, 3], resize_size, interpolation=cv2.INTER_LINEAR)
                img = np.concatenate([rgb_resized, fourth_resized[:, :, np.newaxis]], axis=2)
            elif self.model_channels == 6:
                # 6:RGB
                rgb1_resized = cv2.resize(img[:, :, :3], resize_size, interpolation=cv2.INTER_LINEAR)
                rgb2_resized = cv2.resize(img[:, :, 3:6], resize_size, interpolation=cv2.INTER_LINEAR)
                img = np.concatenate([rgb1_resized, rgb2_resized], axis=2)
            else:
                # :resize
                img = cv2.resize(img, resize_size, interpolation=cv2.INTER_LINEAR)

        # centercrop
        for transform in self.transforms.transforms:
            if isinstance(transform, T.CenterCrop):
                crop_size = transform.size
                if isinstance(crop_size, int):
                    crop_size = (crop_size, crop_size)

                h, w = img.shape[:2]
                if h >= crop_size[0] and w >= crop_size[1]:
                    top = (h - crop_size[0]) // 2
                    left = (w - crop_size[1]) // 2
                    img = img[top:top + crop_size[0], left:left + crop_size[1]]
                break

        return img

    def _apply_normalization(self, tensor):
        """"""
        # 
        for transform in self.transforms.transforms:
            if isinstance(transform, T.Normalize):
                mean = transform.mean
                std = transform.std

                # 
                tensor_channels = tensor.shape[0]

                if tensor_channels == len(mean):
                    # ,
                    for t, m, s in zip(tensor, mean, std):
                        t.sub_(m).div_(s)
                elif len(mean) == 3 and tensor_channels != 3:
                    # 3
                    if tensor_channels == 1:
                        # 1:
                        avg_mean = sum(mean) / 3.0
                        avg_std = sum(std) / 3.0
                        tensor.sub_(avg_mean).div_(avg_std)
                    elif tensor_channels == 4:
                        # 4:3RGB,4
                        for i in range(3):
                            tensor[i].sub_(mean[i]).div_(std[i])
                        tensor[3].sub_(0.0).div_(1.0)
                    elif tensor_channels == 6:
                        # 6:RGB
                        for i in range(3):
                            tensor[i].sub_(mean[i]).div_(std[i])  # RGB
                            tensor[i + 3].sub_(mean[i]).div_(std[i])  # RGB
                else:
                    # :
                    for i in range(tensor_channels):
                        if i < len(mean):
                            tensor[i].sub_(mean[i]).div_(std[i])
                        else:
                            tensor[i].sub_(0.0).div_(1.0)
                break

        return tensor

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        preds = preds[0] if isinstance(preds, (list, tuple)) else preds

        # 
        processed_orig_imgs = []
        for orig_img in orig_imgs:
            # (>3),3
            if len(orig_img.shape) == 3 and orig_img.shape[2] > 3:
                if orig_img.shape[2] == 4:
                    # 4:3
                    vis_img = orig_img[:, :, :3]
                elif orig_img.shape[2] == 6:
                    # 6:RGB
                    vis_img = orig_img[:, :, :3]
                else:
                    # :3
                    vis_img = orig_img[:, :, :3]
                processed_orig_imgs.append(vis_img)
            else:
                processed_orig_imgs.append(orig_img)

        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, processed_orig_imgs, self.batch[0])
        ]