import numpy as np
import  sys
import cv2
from PIL import Image, ImageEnhance

class State():
    def __init__(self, size, move_range):
        self.image = np.zeros(size, dtype=np.float32)
        self.move_range = move_range

    def reset(self, x, noise):
        self.image = x + noise

    def step(self, act):
        neutral = (self.move_range - 1) / 2
        move = act.astype(np.float32)
        move = (move - neutral) / 255
        moved_image = self.image + move[:,np.newaxis,:,:]
        action = np.zeros(self.image.shape[2:], self.image.dtype)

        b, c, h, w = self.image.shape
        for i in range(0,b):
            pil_image = Image.fromarray((self.image[i,0]*255).astype(np.int8)).convert('L')
            contrast_enhancer = ImageEnhance.Contrast(pil_image)
            sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
            brightness_enhancer = ImageEnhance.Brightness(pil_image)

            if np.sum(act[i] == self.move_range) > 0:
                action = cv2.GaussianBlur(self.image[i,0], ksize=(5,5), sigmaX=0.5)
                moved_image[i,0] = np.where(act[i]==self.move_range, action, moved_image[i,0])
            if np.sum(act[i] == self.move_range+1) > 0:
                action = cv2.GaussianBlur(self.image[i,0], ksize=(5,5), sigmaX=1.5)
                moved_image[i,0] = np.where(act[i]==self.move_range+1, action, moved_image[i,0])
            if np.sum(act[i] == self.move_range+2) > 0:
                action = cv2.bilateralFilter(self.image[i,0], d=5, sigmaColor=0.1, sigmaSpace=5)
                moved_image[i,0] = np.where(act[i]==self.move_range+2, action, moved_image[i,0])
            if np.sum(act[i] == self.move_range+3) > 0:
                action = cv2.bilateralFilter(self.image[i,0], d=5, sigmaColor=1.0, sigmaSpace=5)
                moved_image[i,0] = np.where(act[i]==self.move_range+3, action, moved_image[i,0])
            if np.sum(act[i] == self.move_range+4) > 0:
                action = cv2.medianBlur(self.image[i,0], ksize=5)
                moved_image[i,0] = np.where(act[i]==self.move_range+4, action, moved_image[i,0])
            if np.sum(act[i] == self.move_range+5) > 0:
                action = cv2.boxFilter(self.image[i,0], ddepth=-1, ksize=(5,5))
                moved_image[i,0] = np.where(act[i]==self.move_range+5, action, moved_image[i,0])
            if np.sum(act[i] == self.move_range+6) > 0:
                action = self._pil_to_np(contrast_enhancer.enhance(0.95))
                moved_image[i,0] = np.where(act[i]==self.move_range+6, action, moved_image[i,0])
            if np.sum(act[i] == self.move_range+7) > 0:
                action = self._pil_to_np(contrast_enhancer.enhance(1.05))
                moved_image[i,0] = np.where(act[i]==self.move_range+7, action, moved_image[i,0])
            if np.sum(act[i] == self.move_range+8) > 0:
                action = self._pil_to_np(sharpness_enhancer.enhance(0.95))
                moved_image[i,0] = np.where(act[i]==self.move_range+8, action, moved_image[i,0])
            if np.sum(act[i] == self.move_range+9) > 0:
                action = self._pil_to_np(sharpness_enhancer.enhance(1.05))
                moved_image[i,0] = np.where(act[i]==self.move_range+9, action, moved_image[i,0])
            if np.sum(act[i] == self.move_range+10) > 0:
                action = self._pil_to_np(brightness_enhancer.enhance(0.95))
                moved_image[i,0] = np.where(act[i]==self.move_range+10, action, moved_image[i,0])
            if np.sum(act[i] == self.move_range+11) > 0:
                action = self._pil_to_np(brightness_enhancer.enhance(1.05))
                moved_image[i,0] = np.where(act[i]==self.move_range+11, action, moved_image[i,0])

        self.image = moved_image
        # print("Step")

    def _pil_to_np(self, pil_image):
        image_array = np.array(pil_image)
        image_array = image_array.astype(np.float32) / 255.0
        return image_array[np.newaxis, np.newaxis, :, :]
