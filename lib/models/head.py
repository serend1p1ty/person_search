import torch.nn as nn


class Head(nn.Module):
    """
    Extract the features of region proposals (conv4_4 --> conv5).
    The name of each module is exactly the same as in the original caffe code.
    """

    def __init__(self):
        super(Head, self).__init__()
        # conv4
        self.SpatialConvolution_107 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.BN_108 = nn.BatchNorm2d(256)
        self.ReLU_109 = nn.ReLU(inplace=True)
        self.SpatialConvolution_110 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.BN_111 = nn.BatchNorm2d(256)
        self.ReLU_112 = nn.ReLU(inplace=True)
        self.SpatialConvolution_113 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        self.BN_115 = nn.BatchNorm2d(1024)
        self.ReLU_116 = nn.ReLU(inplace=True)

        self.SpatialConvolution_117 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.BN_118 = nn.BatchNorm2d(256)
        self.ReLU_119 = nn.ReLU(inplace=True)
        self.SpatialConvolution_120 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.BN_121 = nn.BatchNorm2d(256)
        self.ReLU_122 = nn.ReLU(inplace=True)
        self.SpatialConvolution_123 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        self.BN_125 = nn.BatchNorm2d(1024)
        self.ReLU_126 = nn.ReLU(inplace=True)

        self.SpatialConvolution_127 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.BN_128 = nn.BatchNorm2d(256)
        self.ReLU_129 = nn.ReLU(inplace=True)
        self.SpatialConvolution_130 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.BN_131 = nn.BatchNorm2d(256)
        self.ReLU_132 = nn.ReLU(inplace=True)
        self.SpatialConvolution_133 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        self.BN_135 = nn.BatchNorm2d(1024)
        self.ReLU_136 = nn.ReLU(inplace=True)

        # conv5
        self.SpatialConvolution_137 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.BN_138 = nn.BatchNorm2d(512)
        self.ReLU_139 = nn.ReLU(inplace=True)
        self.SpatialConvolution_140 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.BN_141 = nn.BatchNorm2d(512)
        self.ReLU_142 = nn.ReLU(inplace=True)
        self.SpatialConvolution_143 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)
        self.SpatialConvolution_144 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2, padding=0)
        self.BN_146 = nn.BatchNorm2d(2048)
        self.ReLU_147 = nn.ReLU(inplace=True)

        self.SpatialConvolution_148 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        self.BN_149 = nn.BatchNorm2d(512)
        self.ReLU_150 = nn.ReLU(inplace=True)
        self.SpatialConvolution_151 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.BN_152 = nn.BatchNorm2d(512)
        self.ReLU_153 = nn.ReLU(inplace=True)
        self.SpatialConvolution_154 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)
        self.BN_156 = nn.BatchNorm2d(2048)
        self.ReLU_157 = nn.ReLU(inplace=True)

        self.SpatialConvolution_158 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        self.BN_159 = nn.BatchNorm2d(512)
        self.ReLU_160 = nn.ReLU(inplace=True)
        self.SpatialConvolution_161 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.BN_162 = nn.BatchNorm2d(512)
        self.ReLU_163 = nn.ReLU(inplace=True)
        self.SpatialConvolution_164 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)
        self.BN_166 = nn.BatchNorm2d(2048)
        self.ReLU_167 = nn.ReLU(inplace=True)

        self.Pooling_168 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        # conv4
        residual = x
        x = self.SpatialConvolution_107(x)
        x = self.BN_108(x)
        x = self.ReLU_109(x)
        x = self.SpatialConvolution_110(x)
        x = self.BN_111(x)
        x = self.ReLU_112(x)
        x = self.SpatialConvolution_113(x)
        x += residual
        x = self.BN_115(x)
        x = self.ReLU_116(x)

        residual = x
        x = self.SpatialConvolution_117(x)
        x = self.BN_118(x)
        x = self.ReLU_119(x)
        x = self.SpatialConvolution_120(x)
        x = self.BN_121(x)
        x = self.ReLU_122(x)
        x = self.SpatialConvolution_123(x)
        x += residual
        x = self.BN_125(x)
        x = self.ReLU_126(x)

        residual = x
        x = self.SpatialConvolution_127(x)
        x = self.BN_128(x)
        x = self.ReLU_129(x)
        x = self.SpatialConvolution_130(x)
        x = self.BN_131(x)
        x = self.ReLU_132(x)
        x = self.SpatialConvolution_133(x)
        x += residual
        x = self.BN_135(x)
        x = self.ReLU_136(x)

        # conv5
        residual = self.SpatialConvolution_144(x)
        x = self.SpatialConvolution_137(x)
        x = self.BN_138(x)
        x = self.ReLU_139(x)
        x = self.SpatialConvolution_140(x)
        x = self.BN_141(x)
        x = self.ReLU_142(x)
        x = self.SpatialConvolution_143(x)
        x += residual
        x = self.BN_146(x)
        x = self.ReLU_147(x)

        residual = x
        x = self.SpatialConvolution_148(x)
        x = self.BN_149(x)
        x = self.ReLU_150(x)
        x = self.SpatialConvolution_151(x)
        x = self.BN_152(x)
        x = self.ReLU_153(x)
        x = self.SpatialConvolution_154(x)
        x += residual
        x = self.BN_156(x)
        x = self.ReLU_157(x)

        residual = x
        x = self.SpatialConvolution_158(x)
        x = self.BN_159(x)
        x = self.ReLU_160(x)
        x = self.SpatialConvolution_161(x)
        x = self.BN_162(x)
        x = self.ReLU_163(x)
        x = self.SpatialConvolution_164(x)
        x += residual
        x = self.BN_166(x)
        x = self.ReLU_167(x)

        x = self.Pooling_168(x)
        return x
