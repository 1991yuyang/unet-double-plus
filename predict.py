import torch as t
from torch import nn
from model import UnetDoublePlus
import os
import json
import cv2
from copy import deepcopy
import numpy as np
import shutil


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
conf = json.load(open("conf.json", "r", encoding="utf-8"))
predict_conf = conf["predict"]
data_dir = predict_conf["data_dir"]
img_suffix = predict_conf["img_suffix"]
img_size = predict_conf["img_size"]
use_best_model = predict_conf["use_best_model"]
num_classes = predict_conf["num_classes"]
backbone_type = predict_conf["backbone_type"]
is_deconv = predict_conf["is_deconv"]
downsample_use_pool = predict_conf["downsample_use_pool"]
is_deep_sup = predict_conf["is_deep_sup"]
save_result = predict_conf["save_result"]
show_result = predict_conf["show_result"]
softmax_op = nn.Softmax(dim=1)


def load_model():
    model = UnetDoublePlus(backbone_type=backbone_type, is_deconv=is_deconv, num_classes=num_classes, downsample_use_pool=downsample_use_pool, is_deep_sup=is_deep_sup)
    model = nn.DataParallel(module=model, device_ids=[0])
    if use_best_model:
        model.load_state_dict(t.load("best.pth"))
    else:
        model.load_state_dict(t.load("epoch.pth"))
    model = model.cuda(0)
    model.eval()
    return model


def load_one_image(img_pth):
    rgb_img = cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB)
    original_h, original_w = rgb_img.shape[:2]
    original_rgb_img = deepcopy(rgb_img)
    img_tensor = t.tensor(np.transpose(cv2.resize(rgb_img, (img_size, img_size)) / 255, axes=[2, 0, 1])).unsqueeze(0).type(t.FloatTensor).cuda(0)
    return img_tensor, original_rgb_img, original_h, original_w


def predict_one_img(img_tensor, model, original_h, original_w):
    with t.no_grad():
        outputs = model(img_tensor)
        if is_deep_sup:
            result = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4
        else:
            result = outputs[3]
        predict_result = t.argmax(softmax_op(result), dim=1)[0].cpu().detach().numpy().astype(np.uint8)
        _, predict_mask = cv2.threshold(cv2.resize(predict_result * 255, (original_w, original_h), cv2.INTER_LINEAR), 0, 255, cv2.THRESH_BINARY)
    return predict_mask


def fusion(predict_mask, original_rgb_img):
    original_bgr_img = cv2.cvtColor(original_rgb_img, cv2.COLOR_BGR2RGB)
    original_bgr_img[:, :, 2][predict_mask == 255] = 255
    return original_bgr_img


def main():
    model = load_model()
    if save_result and os.path.exists("predict_result"):
        shutil.rmtree("predict_result")
    os.mkdir("predict_result")
    for name in os.listdir(data_dir):
        img_pth = os.path.join(data_dir, name)
        img_tensor, original_rgb_img, original_h, original_w = load_one_image(img_pth)
        predict_mask = predict_one_img(img_tensor, model, original_h, original_w)
        fusion_result = fusion(predict_mask, original_rgb_img)
        if show_result:
            cv2.imshow("fusion", cv2.resize(fusion_result, (600, 600)))
            cv2.waitKey()
        if save_result:
            cv2.imwrite("predict_result/%s" % (name.replace(img_suffix, "png"),), fusion_result)


if __name__ == "__main__":
    main()