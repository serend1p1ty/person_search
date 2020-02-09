import torch.nn as nn


class Backbone(nn.Module):
    """
    Extract the basic features of the images (conv1 --> conv4_3).
    The name of each module is exactly the same as in the original caffe code.
    """

    def __init__(self):
        super(Backbone, self).__init__()
        # conv1
        self.SpatialConvolution_0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.BN_1 = nn.BatchNorm2d(64)
        self.ReLU_2 = nn.ReLU(inplace=True)
        self.Pooling_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # conv2
        self.SpatialConvolution_4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.BN_5 = nn.BatchNorm2d(64)
        self.ReLU_6 = nn.ReLU(inplace=True)
        self.SpatialConvolution_7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.BN_8 = nn.BatchNorm2d(64)
        self.ReLU_9 = nn.ReLU(inplace=True)
        self.SpatialConvolution_10 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.SpatialConvolution_11 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.BN_13 = nn.BatchNorm2d(256)
        self.ReLU_14 = nn.ReLU(inplace=True)

        self.SpatialConvolution_15 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.BN_16 = nn.BatchNorm2d(64)
        self.ReLU_17 = nn.ReLU(inplace=True)
        self.SpatialConvolution_18 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.BN_19 = nn.BatchNorm2d(64)
        self.ReLU_20 = nn.ReLU(inplace=True)
        self.SpatialConvolution_21 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.BN_23 = nn.BatchNorm2d(256)
        self.ReLU_24 = nn.ReLU(inplace=True)

        self.SpatialConvolution_25 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.BN_26 = nn.BatchNorm2d(64)
        self.ReLU_27 = nn.ReLU(inplace=True)
        self.SpatialConvolution_28 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.BN_29 = nn.BatchNorm2d(64)
        self.ReLU_30 = nn.ReLU(inplace=True)
        self.SpatialConvolution_31 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.BN_33 = nn.BatchNorm2d(256)
        self.ReLU_34 = nn.ReLU(inplace=True)

        # conv3
        self.SpatialConvolution_35 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.BN_36 = nn.BatchNorm2d(128)
        self.ReLU_37 = nn.ReLU(inplace=True)
        self.SpatialConvolution_38 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.BN_39 = nn.BatchNorm2d(128)
        self.ReLU_40 = nn.ReLU(inplace=True)
        self.SpatialConvolution_41 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.SpatialConvolution_42 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)
        self.BN_44 = nn.BatchNorm2d(512)
        self.ReLU_45 = nn.ReLU(inplace=True)

        self.SpatialConvolution_46 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.BN_47 = nn.BatchNorm2d(128)
        self.ReLU_48 = nn.ReLU(inplace=True)
        self.SpatialConvolution_49 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.BN_50 = nn.BatchNorm2d(128)
        self.ReLU_51 = nn.ReLU(inplace=True)
        self.SpatialConvolution_52 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.BN_54 = nn.BatchNorm2d(512)
        self.ReLU_55 = nn.ReLU(inplace=True)

        self.SpatialConvolution_56 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.BN_57 = nn.BatchNorm2d(128)
        self.ReLU_58 = nn.ReLU(inplace=True)
        self.SpatialConvolution_59 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.BN_60 = nn.BatchNorm2d(128)
        self.ReLU_61 = nn.ReLU(inplace=True)
        self.SpatialConvolution_62 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.BN_64 = nn.BatchNorm2d(512)
        self.ReLU_65 = nn.ReLU(inplace=True)

        self.SpatialConvolution_66 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.BN_67 = nn.BatchNorm2d(128)
        self.ReLU_68 = nn.ReLU(inplace=True)
        self.SpatialConvolution_69 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.BN_70 = nn.BatchNorm2d(128)
        self.ReLU_71 = nn.ReLU(inplace=True)
        self.SpatialConvolution_72 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.BN_74 = nn.BatchNorm2d(512)
        self.ReLU_75 = nn.ReLU(inplace=True)

        # conv4
        self.SpatialConvolution_76 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.BN_77 = nn.BatchNorm2d(256)
        self.ReLU_78 = nn.ReLU(inplace=True)
        self.SpatialConvolution_79 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.BN_80 = nn.BatchNorm2d(256)
        self.ReLU_81 = nn.ReLU(inplace=True)
        self.SpatialConvolution_82 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        self.SpatialConvolution_83 = nn.Conv2d(512, 1024, kernel_size=1, stride=2, padding=0)
        self.BN_85 = nn.BatchNorm2d(1024)
        self.ReLU_86 = nn.ReLU(inplace=True)

        self.SpatialConvolution_87 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.BN_88 = nn.BatchNorm2d(256)
        self.ReLU_89 = nn.ReLU(inplace=True)
        self.SpatialConvolution_90 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.BN_91 = nn.BatchNorm2d(256)
        self.ReLU_92 = nn.ReLU(inplace=True)
        self.SpatialConvolution_93 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        self.BN_95 = nn.BatchNorm2d(1024)
        self.ReLU_96 = nn.ReLU(inplace=True)

        self.SpatialConvolution_97 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.BN_98 = nn.BatchNorm2d(256)
        self.ReLU_99 = nn.ReLU(inplace=True)
        self.SpatialConvolution_100 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.BN_101 = nn.BatchNorm2d(256)
        self.ReLU_102 = nn.ReLU(inplace=True)
        self.SpatialConvolution_103 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        self.BN_105 = nn.BatchNorm2d(1024)
        self.ReLU_106 = nn.ReLU(inplace=True)

    def forward(self, x):
        # conv1
        residual = x
        x = self.SpatialConvolution_0(x)
        x = self.BN_1(x)
        x = self.ReLU_2(x)
        x = self.Pooling_3(x)

        # conv2
        residual = self.SpatialConvolution_11(x)
        x = self.SpatialConvolution_4(x)
        x = self.BN_5(x)
        x = self.ReLU_6(x)
        x = self.SpatialConvolution_7(x)
        x = self.BN_8(x)
        x = self.ReLU_9(x)
        x = self.SpatialConvolution_10(x)
        x += residual
        x = self.BN_13(x)
        x = self.ReLU_14(x)

        residual = x
        x = self.SpatialConvolution_15(x)
        x = self.BN_16(x)
        x = self.ReLU_17(x)
        x = self.SpatialConvolution_18(x)
        x = self.BN_19(x)
        x = self.ReLU_20(x)
        x = self.SpatialConvolution_21(x)
        x += residual
        x = self.BN_23(x)
        x = self.ReLU_24(x)

        residual = x
        x = self.SpatialConvolution_25(x)
        x = self.BN_26(x)
        x = self.ReLU_27(x)
        x = self.SpatialConvolution_28(x)
        x = self.BN_29(x)
        x = self.ReLU_30(x)
        x = self.SpatialConvolution_31(x)
        x += residual
        x = self.BN_33(x)
        x = self.ReLU_34(x)

        # conv3
        residual = self.SpatialConvolution_42(x)
        x = self.SpatialConvolution_35(x)
        x = self.BN_36(x)
        x = self.ReLU_37(x)
        x = self.SpatialConvolution_38(x)
        x = self.BN_39(x)
        x = self.ReLU_40(x)
        x = self.SpatialConvolution_41(x)
        x += residual
        x = self.BN_44(x)
        x = self.ReLU_45(x)

        residual = x
        x = self.SpatialConvolution_46(x)
        x = self.BN_47(x)
        x = self.ReLU_48(x)
        x = self.SpatialConvolution_49(x)
        x = self.BN_50(x)
        x = self.ReLU_51(x)
        x = self.SpatialConvolution_52(x)
        x += residual
        x = self.BN_54(x)
        x = self.ReLU_55(x)

        residual = x
        x = self.SpatialConvolution_56(x)
        x = self.BN_57(x)
        x = self.ReLU_58(x)
        x = self.SpatialConvolution_59(x)
        x = self.BN_60(x)
        x = self.ReLU_61(x)
        x = self.SpatialConvolution_62(x)
        x += residual
        x = self.BN_64(x)
        x = self.ReLU_65(x)

        residual = x
        x = self.SpatialConvolution_66(x)
        x = self.BN_67(x)
        x = self.ReLU_68(x)
        x = self.SpatialConvolution_69(x)
        x = self.BN_70(x)
        x = self.ReLU_71(x)
        x = self.SpatialConvolution_72(x)
        x += residual
        x = self.BN_74(x)
        x = self.ReLU_75(x)

        # conv4
        residual = self.SpatialConvolution_83(x)
        x = self.SpatialConvolution_76(x)
        x = self.BN_77(x)
        x = self.ReLU_78(x)
        x = self.SpatialConvolution_79(x)
        x = self.BN_80(x)
        x = self.ReLU_81(x)
        x = self.SpatialConvolution_82(x)
        x += residual
        x = self.BN_85(x)
        x = self.ReLU_86(x)

        residual = x
        x = self.SpatialConvolution_87(x)
        x = self.BN_88(x)
        x = self.ReLU_89(x)
        x = self.SpatialConvolution_90(x)
        x = self.BN_91(x)
        x = self.ReLU_92(x)
        x = self.SpatialConvolution_93(x)
        x += residual
        x = self.BN_95(x)
        x = self.ReLU_96(x)

        residual = x
        x = self.SpatialConvolution_97(x)
        x = self.BN_98(x)
        x = self.ReLU_99(x)
        x = self.SpatialConvolution_100(x)
        x = self.BN_101(x)
        x = self.ReLU_102(x)
        x = self.SpatialConvolution_103(x)
        x += residual
        x = self.BN_105(x)
        x = self.ReLU_106(x)

        return x
