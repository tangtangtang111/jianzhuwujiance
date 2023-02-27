import torch
import torch.nn as nn
import torch.nn.functional as F

from base import modules as md
'''
1. 深度可分离卷积
'''
class DSC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DSC, self).__init__()
        self.dsc_module = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )

    def forward(self, x):
        return self.dsc_module(x)

'''   
2.加入深度可分离卷积的 组合
'''
class zConv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = DSC(in_channels,out_channels)
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(zConv2dReLU, self).__init__(conv, bn, relu)

'''
3.标准卷积+深度可分离卷积
'''
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        
        # 标准卷积
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        
        '''
        # 深度可分离卷积
        self.conv2 = zConv2dReLU(
            out_channels,
            out_channels,
            use_batchnorm=use_batchnorm,
        )
        '''
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

'''
1.自注意力 MHSA
'''
class MHSA(nn.Module):
    def __init__(self, n_dims, width, height, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        # 2 256 64 64
 
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1) # 2 4 64 4096
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1) # 2 4 64 4096
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)# 2 4 64 4096

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k) # 2 4 4096 4096

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2) #2 4 196 64
        content_position = torch.matmul(content_position, q) # 2 4 196 4096

     
        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out
'''
2.加入 MHSA 的组合
'''
class MConv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            w,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = MHSA(in_channels, w, w)
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(in_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(in_channels)

        else:
            bn = nn.Identity()

        super(MConv2dReLU, self).__init__(conv, bn, relu)


'''
3.标准卷积+MHSA
'''
class MDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            w,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        
        # MHSA卷积
        self.conv2 = MConv2dReLU(
            out_channels,
            w,
            use_batchnorm=use_batchnorm,
        )
        
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

        
class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetPlusPlusDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=True,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx+1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx+1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx+1-depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                if depth_idx == 0 and layer_idx == 0 :
                    blocks[f'x_{depth_idx}_{layer_idx}'] = MDecoderBlock(in_ch, skip_ch, out_ch, 32, **kwargs)
                
                else:
                    blocks[f'x_{depth_idx}_{layer_idx}'] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f'x_{0}_{len(self.in_channels)-1}'] =\
            DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1], **kwargs)
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels)-1):
            for depth_idx in range(self.depth-layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f'x_{depth_idx}_{depth_idx}'](features[depth_idx], features[depth_idx+1])
                    dense_x[f'x_{depth_idx}_{depth_idx}'] = output
                    
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f'x_{idx}_{dense_l_i}'] for idx in range(depth_idx+1, dense_l_i+1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i+1]], dim=1)
                    dense_x[f'x_{depth_idx}_{dense_l_i}'] =\
                        self.blocks[f'x_{depth_idx}_{dense_l_i}'](dense_x[f'x_{depth_idx}_{dense_l_i-1}'], cat_features)
        dense_x[f'x_{0}_{self.depth}'] = self.blocks[f'x_{0}_{self.depth}'](dense_x[f'x_{0}_{self.depth-1}'])
        
        return dense_x[f'x_{0}_{self.depth}']
