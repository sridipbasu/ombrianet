import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down1 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(256, 512)
        self.drop4 = nn.Dropout(0.3)
        self.pool4 = nn.MaxPool2d(2)
        
        self.down4 = DoubleConv(512, 1024)
        self.drop5 = nn.Dropout(0.3)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = nn.Conv2d(1024, 512, kernel_size=2, padding=0)  # matching Conv2D(512, 2)
        self.up_dc1 = DoubleConv(1024, 512)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = nn.Conv2d(512, 256, kernel_size=2, padding=0)
        self.up_dc2 = DoubleConv(512, 256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv3 = nn.Conv2d(256, 128, kernel_size=2, padding=0)
        self.up_dc3 = DoubleConv(256, 128)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv4 = nn.Conv2d(128, 64, kernel_size=2, padding=0)
        self.up_dc4 = DoubleConv(128, 64)
        
        self.final_conv1 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # We need custom padding for Conv2d(kernel_size=2) to match Keras 'same' padding
        # In Keras, same padding with stride 1 and kernel 2 pads 1 pixel on the right/bottom
        def pad_same_k2(x):
            return nn.functional.pad(x, (0, 1, 0, 1))

        x1 = self.inc(x)
        p1 = self.pool1(x1)
        x2 = self.down1(p1)
        p2 = self.pool2(x2)
        x3 = self.down2(p2)
        p3 = self.pool3(x3)
        x4 = self.down3(p3)
        d4 = self.drop4(x4)
        p4 = self.pool4(d4)
        
        x5 = self.down4(p4)
        d5 = self.drop5(x5)
        
        u6 = self.up_conv1(pad_same_k2(self.up1(d5)))
        m6 = torch.cat([d4, u6], dim=1)
        x6 = self.up_dc1(m6)
        
        u7 = self.up_conv2(pad_same_k2(self.up2(x6)))
        m7 = torch.cat([x3, u7], dim=1)
        x7 = self.up_dc2(m7)
        
        u8 = self.up_conv3(pad_same_k2(self.up3(x7)))
        m8 = torch.cat([x2, u8], dim=1)
        x8 = self.up_dc3(m8)
        
        u9 = self.up_conv4(pad_same_k2(self.up4(x8)))
        m9 = torch.cat([x1, u9], dim=1)
        x9 = self.up_dc4(m9)
        
        c9 = self.final_conv1(x9)
        c10 = self.final_conv2(c9)
        return self.sigmoid(c10)

class BitemporalOmbriaNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # Branch 1 (AFTER)
        self.b1_conv1 = DoubleConv(in_channels, 64)
        self.b1_pool1 = nn.MaxPool2d(2)
        self.b1_conv2 = DoubleConv(64, 128)
        self.b1_pool2 = nn.MaxPool2d(2)
        self.b1_conv3 = DoubleConv(128, 256)
        self.b1_pool3 = nn.MaxPool2d(2)
        
        # Branch 2 (BEFORE)
        self.b2_conv1 = DoubleConv(in_channels, 64)
        self.b2_pool1 = nn.MaxPool2d(2)
        self.b2_conv2 = DoubleConv(64, 128)
        self.b2_pool2 = nn.MaxPool2d(2)
        self.b2_conv3 = DoubleConv(128, 256)
        self.b2_pool3 = nn.MaxPool2d(2)
        
        # Bottleneck & Decoder
        self.bottleneck = DoubleConv(512, 512)
        self.drop = nn.Dropout(0.2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = nn.Conv2d(512, 256, kernel_size=2, padding=0)
        self.dec1 = DoubleConv(768, 256)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = nn.Conv2d(256, 128, kernel_size=2, padding=0)
        self.dec2 = DoubleConv(384, 128)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv3 = nn.Conv2d(128, 64, kernel_size=2, padding=0)
        self.dec3 = DoubleConv(192, 64)
        
        self.final_conv1 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        def pad_same_k2(x):
            return nn.functional.pad(x, (0, 1, 0, 1))

        # Branch 1
        x1_1 = self.b1_conv1(x1)
        p1_1 = self.b1_pool1(x1_1)
        x1_2 = self.b1_conv2(p1_1)
        p1_2 = self.b1_pool2(x1_2)
        x1_3 = self.b1_conv3(p1_2)
        p1_3 = self.b1_pool3(x1_3)
        
        # Branch 2
        x2_1 = self.b2_conv1(x2)
        p2_1 = self.b2_pool1(x2_1)
        x2_2 = self.b2_conv2(p2_1)
        p2_2 = self.b2_pool2(x2_2)
        x2_3 = self.b2_conv3(p2_2)
        p2_3 = self.b2_pool3(x2_3)
        
        # Bottleneck Concat
        merged = torch.cat([p1_3, p2_3], dim=1)
        bn = self.drop(self.bottleneck(merged))
        
        # Decoder
        u1 = self.up_conv1(pad_same_k2(self.up1(bn)))
        m1 = torch.cat([x1_3, x2_3, u1], dim=1)
        d1 = self.dec1(m1)
        
        u2 = self.up_conv2(pad_same_k2(self.up2(d1)))
        m2 = torch.cat([x1_2, x2_2, u2], dim=1)
        d2 = self.dec2(m2)
        
        u3 = self.up_conv3(pad_same_k2(self.up3(d2)))
        m3 = torch.cat([x1_1, x2_1, u3], dim=1)
        d3 = self.dec3(m3)
        
        c9 = self.final_conv1(d3)
        c10 = self.final_conv2(c9)
        return self.sigmoid(c10)

class MultimodalOmbriaNet(nn.Module):
    def __init__(self, in_ch_s2=3, in_ch_s1=1):
        super().__init__()
        # Branch 1: S2 AFTER (3 channels)
        self.b1_conv1 = DoubleConv(in_ch_s2, 64)
        self.b1_pool1 = nn.MaxPool2d(2)
        self.b1_conv2 = DoubleConv(64, 128)
        self.b1_pool2 = nn.MaxPool2d(2)
        self.b1_conv3 = DoubleConv(128, 256)
        self.b1_pool3 = nn.MaxPool2d(2)
        
        # Branch 2: S2 BEFORE (3 channels)
        self.b2_conv1 = DoubleConv(in_ch_s2, 64)
        self.b2_pool1 = nn.MaxPool2d(2)
        self.b2_conv2 = DoubleConv(64, 128)
        self.b2_pool2 = nn.MaxPool2d(2)
        self.b2_conv3 = DoubleConv(128, 256)
        self.b2_pool3 = nn.MaxPool2d(2)
        
        # Branch 3: S1 AFTER (1 channel)
        self.b3_conv1 = DoubleConv(in_ch_s1, 64)
        self.b3_pool1 = nn.MaxPool2d(2)
        self.b3_conv2 = DoubleConv(64, 128)
        self.b3_pool2 = nn.MaxPool2d(2)
        self.b3_conv3 = DoubleConv(128, 256)
        self.b3_pool3 = nn.MaxPool2d(2)
        
        # Branch 4: S1 BEFORE (1 channel)
        self.b4_conv1 = DoubleConv(in_ch_s1, 64)
        self.b4_pool1 = nn.MaxPool2d(2)
        self.b4_conv2 = DoubleConv(64, 128)
        self.b4_pool2 = nn.MaxPool2d(2)
        self.b4_conv3 = DoubleConv(128, 256)
        self.b4_pool3 = nn.MaxPool2d(2)
        
        # Bottleneck & Decoder
        # Input to bottleneck is concat of all 4 branches: 256 * 4 = 1024 channels
        self.bottleneck = DoubleConv(1024, 512)
        self.drop = nn.Dropout(0.2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = nn.Conv2d(512, 512, kernel_size=2, padding=0)  # Keras: up6 = Conv2D(512, 2)
        self.dec1 = DoubleConv(1536, 256)  # Keras: merge7 = concatenate([conv3, S1conv3, conv3_2, S1conv3_2, up6], axis=3) -> 256*4 + 512 = 1536
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = nn.Conv2d(256, 128, kernel_size=2, padding=0)
        self.dec2 = DoubleConv(640, 128)   # Keras: merge8 = 128*4 + 128 = 640
        
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv3 = nn.Conv2d(128, 64, kernel_size=2, padding=0)
        self.dec3 = DoubleConv(320, 64)    # Keras: merge9 = 64*4 + 64 = 320
        
        self.final_conv1 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4):
        def pad_same_k2(x):
            return nn.functional.pad(x, (0, 1, 0, 1))

        # Branch 1: S2 AFTER
        x1_1 = self.b1_conv1(x1)
        p1_1 = self.b1_pool1(x1_1)
        x1_2 = self.b1_conv2(p1_1)
        p1_2 = self.b1_pool2(x1_2)
        x1_3 = self.b1_conv3(p1_2)
        p1_3 = self.b1_pool3(x1_3)
        
        # Branch 2: S2 BEFORE
        x2_1 = self.b2_conv1(x2)
        p2_1 = self.b2_pool1(x2_1)
        x2_2 = self.b2_conv2(p2_1)
        p2_2 = self.b2_pool2(x2_2)
        x2_3 = self.b2_conv3(p2_2)
        p2_3 = self.b2_pool3(x2_3)
        
        # Branch 3: S1 AFTER
        x3_1 = self.b3_conv1(x3)
        p3_1 = self.b3_pool1(x3_1)
        x3_2 = self.b3_conv2(p3_1)
        p3_2 = self.b3_pool2(x3_2)
        x3_3 = self.b3_conv3(p3_2)
        p3_3 = self.b3_pool3(x3_3)
        
        # Branch 4: S1 BEFORE
        x4_1 = self.b4_conv1(x4)
        p4_1 = self.b4_pool1(x4_1)
        x4_2 = self.b4_conv2(p4_1)
        p4_2 = self.b4_pool2(x4_2)
        x4_3 = self.b4_conv3(p4_2)
        p4_3 = self.b4_pool3(x4_3)
        
        # Bottleneck Concat (all 4 branches)
        merged = torch.cat([p1_3, p2_3, p3_3, p4_3], dim=1)
        bn = self.drop(self.bottleneck(merged))
        
        # Decoder
        u1 = self.up_conv1(pad_same_k2(self.up1(bn)))
        m1 = torch.cat([x1_3, x2_3, x3_3, x4_3, u1], dim=1)
        d1 = self.dec1(m1)
        
        u2 = self.up_conv2(pad_same_k2(self.up2(d1)))
        m2 = torch.cat([x1_2, x2_2, x3_2, x4_2, u2], dim=1)
        d2 = self.dec2(m2)
        
        u3 = self.up_conv3(pad_same_k2(self.up3(d2)))
        m3 = torch.cat([x1_1, x2_1, x3_1, x4_1, u3], dim=1)
        d3 = self.dec3(m3)
        
        c9 = self.final_conv1(d3)
        c10 = self.final_conv2(c9)
        return self.sigmoid(c10)
