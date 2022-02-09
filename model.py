import torch as t
from torch import nn
from torchvision import models


class Conv3X3(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn, act_func):
        super(Conv3X3, self).__init__()
        bias = not is_bn
        if act_func.lower() == "relu":
            act = nn.ReLU
        if act_func.lower() == "sigmoid":
            act = nn.Sigmoid()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        if is_bn:
            self.block.add_module("last", nn.Sequential(
                nn.BatchNorm2d(num_features=out_channels),
                act()
            ))
        else:
            self.block.add_module("last", act())

    def forward(self, x):
        return self.block(x)


class Upsample(nn.Module):

    def __init__(self, in_channels, is_deconv, is_bn, act_func):
        super(Upsample, self).__init__()
        if not is_deconv:
            self.block = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            bias = not is_bn
            if act_func.lower() == "relu":
                act = nn.ReLU
            if act_func.lower() == "sigmoid":
                act = nn.Sigmoid
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2, padding=0, bias=bias)
            )
            if is_bn:
                self.block.add_module("last", nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    act()
                ))
            else:
                self.block.add_module("last", act())

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):

    def __init__(self, backbone_type, downsample_use_pool):
        super(Encoder, self).__init__()
        assert backbone_type.lower() in ["resnet18", "resnet34", "resnet50", "resnet101"]
        if backbone_type.lower() in ["resnet18", "resnet34"]:
            self.out_channels_of_layer0_to_layer5 = [64, 64, 128, 256, 512]
        else:
            self.out_channels_of_layer0_to_layer5 = [64, 256, 512, 1024, 2048]
        if backbone_type.lower() == "resnet18":
            model = models.resnet18(pretrained=True)
        elif backbone_type.lower() == "resnet34":
            model = models.resnet34(pretrained=True)
        elif backbone_type.lower() == "resnet50":
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet101(pretrained=True)

        self.layer0 = nn.Sequential(
            Conv3X3(in_channels=3, out_channels=self.out_channels_of_layer0_to_layer5[0] // 4, is_bn=True, act_func="relu"),
            Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5[0] // 4, out_channels=self.out_channels_of_layer0_to_layer5[0], is_bn=True, act_func="relu")
        )
        if downsample_use_pool:
            self.downsample0 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_channels=self.out_channels_of_layer0_to_layer5[0], out_channels=self.out_channels_of_layer0_to_layer5[1], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=self.out_channels_of_layer0_to_layer5[1]),
                nn.ReLU()
            )
        else:
            self.downsample0 = nn.Sequential(
                model.layer1,
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )

        self.layer1 = nn.Sequential(
            Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5[1], out_channels=self.out_channels_of_layer0_to_layer5[1] // 4, is_bn=True, act_func="relu"),
            Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5[1] // 4, out_channels=self.out_channels_of_layer0_to_layer5[1], is_bn=True, act_func="relu")
        )
        if downsample_use_pool:
            self.downsample1 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_channels=self.out_channels_of_layer0_to_layer5[1], out_channels=self.out_channels_of_layer0_to_layer5[2], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=self.out_channels_of_layer0_to_layer5[2]),
                nn.ReLU()
            )
        else:
            self.downsample1 = model.layer2

        self.layer2 = nn.Sequential(
            Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5[2], out_channels=self.out_channels_of_layer0_to_layer5[2] // 4, is_bn=True, act_func="relu"),
            Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5[2] // 4, out_channels=self.out_channels_of_layer0_to_layer5[2], is_bn=True, act_func="relu")
        )
        if downsample_use_pool:
            self.downsample2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_channels=self.out_channels_of_layer0_to_layer5[2], out_channels=self.out_channels_of_layer0_to_layer5[3], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=self.out_channels_of_layer0_to_layer5[3]),
                nn.ReLU()
            )
        else:
            self.downsample2 = model.layer3

        self.layer3 = nn.Sequential(
            Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5[3], out_channels=self.out_channels_of_layer0_to_layer5[3] // 4, is_bn=True, act_func="relu"),
            Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5[3] // 4, out_channels=self.out_channels_of_layer0_to_layer5[3], is_bn=True, act_func="relu")
        )
        if downsample_use_pool:
            self.downsample3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_channels=self.out_channels_of_layer0_to_layer5[3], out_channels=self.out_channels_of_layer0_to_layer5[4], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=self.out_channels_of_layer0_to_layer5[4]),
                nn.ReLU()
            )
        else:
            self.downsample3 = model.layer4

        self.layer4 = nn.Sequential(
            Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5[4], out_channels=self.out_channels_of_layer0_to_layer5[4] // 4, is_bn=True, act_func="relu"),
            Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5[4] // 4, out_channels=self.out_channels_of_layer0_to_layer5[4], is_bn=True, act_func="relu")
        )

    def forward(self, x):
        layer0_result = self.layer0(x)
        downsample0 = self.downsample0(layer0_result)
        layer1_result = self.layer1(downsample0)
        downsample1 = self.downsample1(layer1_result)
        layer2_result = self.layer2(downsample1)
        downsample2 = self.downsample2(layer2_result)
        layer3_result = self.layer3(downsample2)
        downsample3 = self.downsample3(layer3_result)
        layer4_result = self.layer4(downsample3)
        return [layer0_result, layer1_result, layer2_result, layer3_result, layer4_result]


class Decoder(nn.Module):

    def __init__(self, is_deconv, encoder_backbone_type):
        super(Decoder, self).__init__()
        if encoder_backbone_type.lower() in ["resnet18", "resnet34"]:
            self.out_channels_of_layer0_to_layer5_of_encoder = [64, 64, 128, 256, 512]
        else:
            self.out_channels_of_layer0_to_layer5_of_encoder = [64, 256, 512, 1024, 2048]

        self.decoder1_up1 = Upsample(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], is_deconv=is_deconv, is_bn=True, act_func="relu")
        self.decoder1_conv1 = nn.Sequential(
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0] + self.out_channels_of_layer0_to_layer5_of_encoder[1], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0], is_bn=True, act_func="relu"),
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0], is_bn=True, act_func="relu")
            )

        self.decoder2_up1 = Upsample(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[2], is_deconv=is_deconv, is_bn=True, act_func="relu")
        self.decoder2_conv1 = nn.Sequential(
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1] + self.out_channels_of_layer0_to_layer5_of_encoder[2], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], is_bn=True, act_func="relu"),
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], is_bn=True, act_func="relu")
            )
        self.decoder2_up2 = Upsample(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], is_deconv=is_deconv, is_bn=True, act_func="relu")
        self.decoder2_conv2 = nn.Sequential(
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0] * 2 + self.out_channels_of_layer0_to_layer5_of_encoder[1], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0], is_bn=True, act_func="relu"),
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0], is_bn=True, act_func="relu")
            )

        self.decoder3_up1 = Upsample(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[3], is_deconv=is_deconv, is_bn=True, act_func="relu")
        self.decoder3_conv1 = nn.Sequential(
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[2] + self.out_channels_of_layer0_to_layer5_of_encoder[3], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[2], is_bn=True, act_func="relu"),
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[2], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[2], is_bn=True, act_func="relu")
            )
        self.decoder3_up2 = Upsample(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[2], is_deconv=is_deconv, is_bn=True, act_func="relu")
        self.decoder3_conv2 = nn.Sequential(
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1] * 2 + self.out_channels_of_layer0_to_layer5_of_encoder[2], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], is_bn=True, act_func="relu"),
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], is_bn=True, act_func="relu")
            )
        self.decoder3_up3 = Upsample(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], is_deconv=is_deconv, is_bn=True, act_func="relu")
        self.decoder3_conv3 = nn.Sequential(
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0] * 3 + self.out_channels_of_layer0_to_layer5_of_encoder[1], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0], is_bn=True, act_func="relu"),
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0], is_bn=True, act_func="relu")
            )

        self.decoder4_up1 = Upsample(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[4], is_deconv=is_deconv, is_bn=True, act_func="relu")
        self.decoder4_conv1 = nn.Sequential(
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[3] + self.out_channels_of_layer0_to_layer5_of_encoder[4], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[3], is_bn=True, act_func="relu"),
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[3], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[3], is_bn=True, act_func="relu")
            )
        self.decoder4_up2 = Upsample(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[3], is_deconv=is_deconv, is_bn=True, act_func="relu")
        self.decoder4_conv2 = nn.Sequential(
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[2] * 2 + self.out_channels_of_layer0_to_layer5_of_encoder[3], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[2], is_bn=True, act_func="relu"),
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[2], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[2], is_bn=True, act_func="relu")
            )
        self.decoder4_up3 = Upsample(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[2], is_deconv=is_deconv, is_bn=True, act_func="relu")
        self.decoder4_conv3 = nn.Sequential(
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1] * 3 + self.out_channels_of_layer0_to_layer5_of_encoder[2], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], is_bn=True, act_func="relu"),
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], is_bn=True, act_func="relu")
            )
        self.decoder4_up4 = Upsample(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[1], is_deconv=is_deconv, is_bn=True, act_func="relu")
        self.decoder4_conv4 = nn.Sequential(
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0] * 4 + self.out_channels_of_layer0_to_layer5_of_encoder[1], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0], is_bn=True, act_func="relu"),
                Conv3X3(in_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0], out_channels=self.out_channels_of_layer0_to_layer5_of_encoder[0], is_bn=True, act_func="relu")
            )

    def forward(self, encoder_outputs):
        decoder1_up1_result = self.decoder1_up1(encoder_outputs[1])
        decoder1_conv1_result = self.decoder1_conv1(t.cat([encoder_outputs[0], decoder1_up1_result], dim=1))

        decoder2_up1_result = self.decoder2_up1(encoder_outputs[2])
        decoder2_conv1_result = self.decoder2_conv1(t.cat([encoder_outputs[1], decoder2_up1_result], dim=1))
        decoder2_up2_result = self.decoder2_up2(decoder2_conv1_result)
        decoder2_conv2_result = self.decoder2_conv2(t.cat([encoder_outputs[0], decoder1_conv1_result, decoder2_up2_result], dim=1))

        decoder3_up1_result = self.decoder3_up1(encoder_outputs[3])
        decoder3_conv1_result = self.decoder3_conv1(t.cat([encoder_outputs[2], decoder3_up1_result], dim=1))
        decoder3_up2_result = self.decoder3_up2(decoder3_conv1_result)
        decoder3_conv2_result = self.decoder3_conv2(t.cat([encoder_outputs[1], decoder2_conv1_result, decoder3_up2_result], dim=1))
        decoder3_up3_result = self.decoder3_up3(decoder3_conv2_result)
        decoder3_conv3_result = self.decoder3_conv3(t.cat([encoder_outputs[0], decoder1_conv1_result, decoder2_conv2_result, decoder3_up3_result], dim=1))

        decoder4_up1_result = self.decoder4_up1(encoder_outputs[4])
        decoder4_conv1_result = self.decoder4_conv1(t.cat([encoder_outputs[3], decoder4_up1_result], dim=1))
        decoder4_up2_result = self.decoder4_up2(decoder4_conv1_result)
        decoder4_conv2_result = self.decoder4_conv2(t.cat([encoder_outputs[2], decoder3_conv1_result, decoder4_up2_result], dim=1))
        decoder4_up3_result = self.decoder4_up3(decoder4_conv2_result)
        decoder4_conv3_result = self.decoder4_conv3(t.cat([encoder_outputs[1], decoder2_conv1_result, decoder3_conv2_result, decoder4_up3_result], dim=1))
        decoder4_up4_result = self.decoder4_up4(decoder4_conv3_result)
        decoder4_conv4_result = self.decoder4_conv4(t.cat([encoder_outputs[0], decoder1_conv1_result, decoder2_conv2_result, decoder3_conv3_result, decoder4_up4_result], dim=1))
        return [decoder1_conv1_result, decoder2_conv2_result, decoder3_conv3_result, decoder4_conv4_result]


class UnetDoublePlus(nn.Module):

    def __init__(self, backbone_type, is_deconv, num_classes, downsample_use_pool, is_deep_sup):
        super(UnetDoublePlus, self).__init__()
        self.is_deep_sup = is_deep_sup
        self.encoder = Encoder(backbone_type, downsample_use_pool)
        self.decoder = Decoder(is_deconv=is_deconv, encoder_backbone_type=backbone_type)
        if is_deep_sup:
            self.head1 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            self.head2 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            self.head3 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.head4 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        encoder_result = self.encoder(x)
        decoder_result = self.decoder(encoder_result)
        decoder4_result = self.head4(decoder_result[3])
        if self.is_deep_sup:
            decoder1_result = self.head1(decoder_result[0])
            decoder2_result = self.head2(decoder_result[1])
            decoder3_result = self.head3(decoder_result[2])
            return decoder1_result, decoder2_result, decoder3_result, decoder4_result
        return None, None, None, decoder4_result


if __name__ == "__main__":
    x = t.randn(2, 3, 64, 64)
    model = UnetDoublePlus(backbone_type="resnet101", is_deconv=False, num_classes=2, downsample_use_pool=True, is_deep_sup=True)
    output = model(x)
    for result in output:
        print(result.size())
