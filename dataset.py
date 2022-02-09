from torch.utils import data
import cv2
import json
import numpy as np
import torch as t
import os
from numpy import random as rd
"""
data_root_dir:
    train_img
        0.jpg
        1.jpg
        .......
    train_label
        0.jpg
        1.jpg
        ......
    valid_img
        0.jpg
        1.jpg
        ......
    valid_label
        0.jpg
        1.jpg
        ......
    test_img
        0.jpg
        1.jpg
        ......
    test_label
        0.jpg
        1.jpg
        ......
"""


class MySet(data.Dataset):

    def __init__(self, label_file_dir, img_file_dir, img_size, is_train, img_suffix, is_json_format, num_classes):
        """

        :param label_file_dir: the dir where label file is stored, pixel value of ground truth mask shold in [0, 1, ..., num_classes - 1]
        :param img_file_dir:  the dir where image file is stored
        :param img_size:  image size that input model, tuple type
        :param is_train:  True indicate training, False indicate evaluation
        :param img_suffix:  "png", "jpg", "bmp", ...
        :param is_json_format:  True indicate label file in label_file_dir is json format, otherwise is mask picture format
        :param num_classes:  number of classes
        """
        self.is_train = is_train
        self.num_classes = num_classes
        self.img_suffix = img_suffix
        self.img_size = tuple(img_size) if isinstance(img_size, tuple) or isinstance(img_size, list) else (img_size, img_size)
        names = [".".join(img_name.split(".")[:-1]) for img_name in os.listdir(img_file_dir)]
        self.img_pths = [os.path.join(img_file_dir, "%s.%s" % (name, img_suffix)) for name in names]
        self.is_json_format = is_json_format
        if is_json_format:
            self.label_pths = [os.path.join(label_file_dir, "%s.%s" % (name, "json")) for name in names]
        else:
            self.label_pths = [os.path.join(label_file_dir, "%s.%s" % (name, img_suffix)) for name in names]

    def __getitem__(self, index):
        img_pth = self.img_pths[index]
        label_pth = self.label_pths[index]
        img = cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB)
        if self.is_json_format:
            label = self.json_to_label_img(label_pth)
        else:
            label = cv2.imread(label_pth, 0)
            if label is None:
                label = cv2.imread(".".join(label_pth.split(".")[:-1] + list({"png", "jpg"} - {self.img_suffix})), 0)
            if self.num_classes == 2 and np.sort(np.unique(label)).tolist() != [0, 1]:
                ret, label = cv2.threshold(label, 0, 255, cv2.THRESH_BINARY)
                label = label // 255
        if self.is_train:
            img, label = self.data_aug(img, label)
            # cv2.imshow("img", img)
            # cv2.imshow("label", label * 255)
            # cv2.waitKey()
        img = cv2.resize(img, self.img_size) / 255
        if self.num_classes == 2 and np.sort(np.unique(label)).tolist() == [0, 1]:
            label = cv2.resize(label * 255, self.img_size, cv2.INTER_LINEAR)
            _, label = cv2.threshold(label, 0, 1, cv2.THRESH_BINARY)
        else:
            label = cv2.resize(label, self.img_size, cv2.INTER_NEAREST)
        # cv2.imshow("img", img)
        # cv2.imshow("label", label * 255)
        # cv2.waitKey()
        return t.tensor(np.transpose(img, axes=[2, 0, 1])).type(t.FloatTensor), t.tensor(label).type(t.LongTensor),

    def __len__(self):
        return len(self.img_pths)

    def json_to_label_img(self, json_file_path):
        """

        :param json_file_path: json file path
        :return: current json file convert to uint8 ndarray label
        """
        with open(json_file_path, 'r') as load_f:
            load_dict = json.load(load_f)
        contours = []
        img_height = load_dict["imageHeight"]
        img_width = load_dict["imageWidth"]
        for shape in load_dict["shapes"]:
            contour = shape["points"]
            contours.append(np.array(contour).reshape((-1, 1, 2)).astype(np.int32))
        label = np.zeros(shape=[img_height, img_width], dtype=np.uint8)
        label = cv2.drawContours(label, contours, -1, 1, -1)
        return label

    def data_aug(self, img, label):
        """

        :param img: opencv image
        :param label: label mask picture
        :return: transformed image and label
        """
        if rd.random() < 0.5:
            angle = rd.randint(-30, 30)
            img = self.random_rotate_img(img, angle)
            label = self.random_rotate_img(label, angle)
        if rd.random() < 0.5:
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)
        if rd.random() < 0.5:
            img = cv2.flip(img, 0)
            label = cv2.flip(label, 0)
        if rd.random() < 0.5:
            img, label = self.random_shift(img, label)
        if rd.random() < 0.5:
            img, label = self.random_scale(img, label)
        return img, label

    def random_rotate_img(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        shuchu = cv2.warpAffine(image, M, (nW, nH))
        return shuchu

    def random_shift(self, img, label):
        rows, cols, channels = img.shape
        x_dist = rd.randint(0, cols // 4)
        y_dist = rd.randint(0, rows // 4)
        M = np.float32([[1, 0, x_dist], [0, 1, y_dist]])
        img_res = cv2.warpAffine(img, M, (cols, rows))
        label_res = cv2.warpAffine(label, M, (cols, rows))
        # cv.warpAffine()第三个参数为输出的图像大小，值得注意的是该参数形式为(width, height)。
        return img_res, label_res

    def random_noise(self, image, mean=0, var=0.001):
        image = np.array(image / 255, dtype=float)  # 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
        noise = np.random.normal(mean, var ** 0.5, image.shape)  # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
        out = image + noise  # 将噪声和原始图像进行相加得到加噪后的图像
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)  # clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
        out = np.uint8(out * 255)  # 解除归一化，乘以255将加噪后的图像的像素值恢复
        # cv.imshow("gasuss", out)
        noise = noise * 255
        return [noise, out]

    def random_scale(self, img, label, img_mean=255):
        # 获取图像的各个维度
        height, width, depth = img.shape
        # 随机缩放尺度
        ratio = rd.uniform(1, 4)
        new_height = int(height * ratio)
        new_width = int(width * ratio)
        left = int(rd.uniform(0, new_width - width))
        top = int(rd.uniform(0, new_height - height))
        # 确定缩放后的图像的维度
        expand_image = np.ones((new_height, new_width, depth), dtype=img.dtype) * img_mean
        expand_image[top: top + height, left: left + width, :] = img
        expand_label = np.zeros((new_height, new_width), dtype=label.dtype)
        expand_label[top:top + height, left:left + width] = label
        return expand_image, expand_label


def make_loader(label_file_dir, img_file_dir, img_size, is_train, img_suffix, batch_size, is_json_format, num_workers, drop_last, num_classes):
    loader = iter(data.DataLoader(MySet(label_file_dir, img_file_dir, img_size, is_train, img_suffix, is_json_format, num_classes), batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers))
    return loader


if __name__ == "__main__":
    json_file_dir = r"/home/yuyang/data/Crack_Segmentation_Dataset/train_label"
    img_file_dir = r"/home/yuyang/data/Crack_Segmentation_Dataset/train_img"
    img_size = (512, 512)

    s = MySet(json_file_dir, img_file_dir, img_size, True, "jpg", False)
    img, label = s[0]
    print(img.size())
    print(label.size())
