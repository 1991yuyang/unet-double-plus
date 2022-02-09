import torch as t
from torch import nn, optim
from dataset import make_loader
from loss import DiceLoss_CeLoss
from metric import SegmentationMetric
import os
import json
from model import UnetDoublePlus
conf = json.load(open("conf.json", "r", encoding="utf-8"))
train_conf = conf["train"]
CUDA_VISIBLE_DEVICES = train_conf["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
device_ids = [int(i) for i in CUDA_VISIBLE_DEVICES.split(",")]
epoch = train_conf["epoch"]
batch_size = train_conf["batch_size"]
init_lr = train_conf["init_lr"]
final_lr = train_conf["final_lr"]
lr_de_range = final_lr / init_lr
lr_de_rate = lr_de_range ** (1 / epoch)
num_classes = train_conf["num_classes"]
is_deconv = train_conf["is_deconv"]
downsample_use_pool = train_conf["downsample_use_pool"]
print_every_step = train_conf["print_every_step"]
decoder1_to_4_loss_weights = train_conf["decoder1_to_4_loss_weights"]
avg_result_loss_weight = train_conf["avg_result_loss_weight"]
save_model_depend_on_iou = train_conf["save_model_depend_on_iou"]
ce_loss_weight = train_conf["ce_loss_weight"]
dice_loss_weight = train_conf["dice_loss_weight"]
data_root_dir = train_conf["data_root_dir"]
backbone_type = train_conf["backbone_type"]
img_suffix = train_conf["img_suffix"]
img_size = train_conf["img_size"]
label_is_json_format = train_conf["label_is_json_format"]
num_workers = train_conf["num_workers"]
weight_decay = train_conf["weight_decay"]
is_deep_sup = train_conf["is_deep_sup"]
best_valid_metric = float("inf")
softmax_op = nn.Softmax(dim=1)
train_img_dir = os.path.join(data_root_dir, "train_img")
train_label_dir = os.path.join(data_root_dir, "train_label")
valid_img_dir = os.path.join(data_root_dir, "valid_img")
valid_label_dir = os.path.join(data_root_dir, "valid_label")


def train_epoch(model, criterion, optimizer, metric_obj, current_epoch, train_loader):
    model.train()
    step = len(train_loader)
    current_step = 1
    for d_train, l_train in train_loader:
        d_train_cuda = d_train.cuda(device_ids[0])
        l_train_cuda = l_train.cuda(device_ids[0])
        decoder_outputs = model(d_train_cuda)
        if is_deep_sup:
            avg_result = (decoder_outputs[0] + decoder_outputs[1] + decoder_outputs[2] + decoder_outputs[3]) / 4
            train_loss = decoder1_to_4_loss_weights[0] * criterion(decoder_outputs[0], l_train_cuda) + \
                         decoder1_to_4_loss_weights[1] * criterion(decoder_outputs[1], l_train_cuda) + \
                         decoder1_to_4_loss_weights[2] * criterion(decoder_outputs[2], l_train_cuda) + \
                         decoder1_to_4_loss_weights[3] * criterion(decoder_outputs[3], l_train_cuda) + \
                         avg_result_loss_weight * criterion(avg_result, l_train_cuda)
        else:
            avg_result = decoder_outputs[3]
            train_loss = criterion(avg_result, l_train_cuda)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_predict_result = t.argmax(softmax_op(avg_result), dim=1)
        pa, cpa, mpa, IoU, mIoU = metric_obj.accum_calc(train_predict_result.cpu().detach(), l_train_cuda.cpu().detach(), [])
        metric_obj.reset()
        if current_step % print_every_step == 0:
            print("epoch:%d/%d, step:%d/%d, train_loss:%.5f, pa:%.5f, cpa:%s, mpa:%.5f, IoU:%s, mIoU:%.5f" % (current_epoch, epoch, current_step, step, train_loss.item(), pa, str(cpa), mpa, str(IoU), mIoU))
        current_step += 1
    print("saving epoch model......")
    t.save(model.state_dict(), "epoch.pth")
    metric_obj.reset()
    return model, metric_obj


def valid_epoch(model, criterion, valid_loader, metric_obj, current_epoch):
    global best_valid_metric
    model.eval()
    valid_losses = 0
    step = len(valid_loader)
    for d_valid, l_valid in valid_loader:
        d_valid_cuda = d_valid.cuda(device_ids[0])
        l_valid_cuda = l_valid.cuda(device_ids[0])
        with t.no_grad():
            decoder_outputs = model(d_valid_cuda)
            if is_deep_sup:
                avg_result = (decoder_outputs[0] + decoder_outputs[1] + decoder_outputs[2] + decoder_outputs[3]) / 4
                valid_loss = decoder1_to_4_loss_weights[0] * criterion(decoder_outputs[0], l_valid_cuda) + \
                             decoder1_to_4_loss_weights[1] * criterion(decoder_outputs[1], l_valid_cuda) + \
                             decoder1_to_4_loss_weights[2] * criterion(decoder_outputs[2], l_valid_cuda) + \
                             decoder1_to_4_loss_weights[3] * criterion(decoder_outputs[3], l_valid_cuda) + \
                             avg_result_loss_weight * criterion(avg_result, l_valid_cuda)
            else:
                avg_result = decoder_outputs[3]
                valid_loss = criterion(avg_result, l_valid_cuda)
            valid_losses += valid_loss.item()
            valid_predict_result = t.argmax(softmax_op(avg_result), dim=1)
            pa, cpa, mpa, IoU, mIoU = metric_obj.accum_calc(valid_predict_result.cpu().detach(), l_valid_cuda.cpu().detach(), [])
    metric_obj.reset()
    valid_loss = valid_losses / step
    if not save_model_depend_on_iou:
        if valid_loss < best_valid_metric:
            best_valid_metric = valid_loss
            print("saving best model......")
            t.save(model.state_dict(), "best.pth")
    else:
        if 1 / mIoU < best_valid_metric:
            best_valid_metric = 1 / mIoU
            print("saving best model......")
            t.save(model.state_dict(), "best.pth")
    print("====================valid epoch:%d=====================" % (current_epoch,))
    print("valid_loss:%.5f, pa:%.5f, cpa:%s, mpa:%.5f, IoU:%s, mIoU:%.5f" % (valid_loss, pa, str(cpa), mpa, str(IoU), mIoU))
    print("=======================================================")
    return model, metric_obj


def main():
    model = UnetDoublePlus(backbone_type=backbone_type, num_classes=num_classes, is_deconv=is_deconv, downsample_use_pool=downsample_use_pool, is_deep_sup=is_deep_sup)
    model = nn.DataParallel(module=model, device_ids=device_ids)
    model = model.cuda(device_ids[0])
    criterion = DiceLoss_CeLoss(dice_loss_weight, ce_loss_weight).cuda(device_ids[0])
    optimizer = optim.Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay)
    lr_sch = optim.lr_scheduler.StepLR(optimizer, 1, lr_de_rate)
    metric_obj = SegmentationMetric(numClass=num_classes)
    for e in range(epoch):
        print("learning rate:%f" % (lr_sch.get_lr()[0],))
        train_loader = make_loader(train_label_dir, train_img_dir, img_size, True, img_suffix, batch_size, label_is_json_format, num_workers, True, num_classes)
        valid_loader = make_loader(valid_label_dir, valid_img_dir, img_size, False, img_suffix, batch_size, label_is_json_format, num_workers, True, num_classes)
        model, metric_obj = train_epoch(model, criterion, optimizer, metric_obj, e + 1, train_loader)
        model, metric_obj = valid_epoch(model, criterion, valid_loader, metric_obj, e + 1)
        lr_sch.step()


if __name__ == "__main__":
    main()