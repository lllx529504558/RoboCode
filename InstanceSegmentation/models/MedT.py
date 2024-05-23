import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AxialBlock_global(nn.Module):
    def __init__(self, in_planes, out_planes, mid_planes=64, groups=8, kernel_size=3, stride=1):
        assert (in_planes % groups == 0) and (out_planes % groups == 0) # in_planes & out_planes 必须是groups的整数倍
        super(AxialBlock_global, self).__init__()
        self.conv_down = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu = nn.ReLU(inplace=True)
        self.height_block = AxialAttention_global(mid_planes, mid_planes, groups=groups, kernel_size=kernel_size, width=False)
        self.width_block = AxialAttention_global(mid_planes, mid_planes, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)
            
    def forward(self, x):
        x_out = self.conv1x1(x)
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.height_block(out)
        out = self.width_block(out)
        out = self.conv_up(out)
        out = self.bn2(out)
        out = torch.add(out, x_out)
        return out

class AxialBlock_partial(nn.Module):
    def __init__(self, in_planes, out_planes, mid_planes=64, groups=8, kernel_size=3, stride=1):
        assert (in_planes % groups == 0) and (out_planes % groups == 0) # in_planes & out_planes 必须是groups的整数倍
        super(AxialBlock_partial, self).__init__()
        self.conv_down = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu = nn.ReLU(inplace=True)
        self.height_block = AxialAttention_partial(mid_planes, mid_planes, groups=groups, kernel_size=kernel_size, width=False)
        self.width_block = AxialAttention_partial(mid_planes, mid_planes, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)
    def forward(self, x):
        x_out = self.conv1x1(x)
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.height_block(out)
        out = self.width_block(out)
        out = self.conv_up(out)
        out = self.bn2(out)
        out = torch.add(out, x_out)
        return out

def qkv_transform(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

class AxialAttention_global(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56, stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0) # in_planes & out_planes 必须是groups的整数倍
        super(AxialAttention_global, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = in_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
    
        # 多头自注意力机制
        self.qkv_transform = qkv_transform(in_planes, out_planes*2, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn_qkv = nn.BatchNorm1d(out_planes*2)
        self.bn_similarity = nn.BatchNorm2d(groups*3)
        self.bn_output = nn.BatchNorm1d(out_planes*2)

        # 门单元参数初始化
        self.f_qr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # 相对位置编码
        self.relative = nn.Parameter(torch.randn(self.group_planes*2, kernel_size*2-1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride)
        self.reset_parameters()
    
    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3) # N, C, H, W -> N, H, C, W
        else:
            x = x.permute(0, 3, 1, 2) # N, C, H, W -> N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N*W, C, H) # 使x在内存中连续，并reshape为(N*W, C, H) -> torch.Size([1280, 32, 128])

        # 多头自注意力机制
        qkv = self.bn_qkv(self.qkv_transform(x)) # torch.Size([1280, 64, 128])
        q, k, v = torch.split(qkv.reshape(N*W, self.groups, self.group_planes*2, H), [self.group_planes//2, self.group_planes//2, self.group_planes], dim=2)
        # 计算相对位置编码
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes*2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes//2, self.group_planes//2, self.group_planes], dim=0)
        # print(q.shape, k.shape, v.shape)
        # print(q_embedding.shape, k_embedding.shape, v_embedding.shape)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci,bgcj->bgij', q, k)
        qr = torch.mul(qr, self.f_qr) # 限制位置信息的重要性
        kr = torch.mul(kr, self.f_kr) # 限制位置信息的重要性
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N*W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=-1)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        sv = torch.mul(sv, self.f_sv) # 限制位置信息的重要性
        sve = torch.mul(sve, self.f_sve) # 限制位置信息的重要性
        stacked_output = torch.cat([sv, sve], dim=1).view(N*W, self.out_planes*2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)
        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)
        if self.stride > 1:
            output = self.pooling(output)
        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0, math.sqrt(1. / self.group_planes))


class AxialAttention_partial(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56, stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0) # in_planes & out_planes 必须是groups的整数倍
        super(AxialAttention_partial, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = in_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
    
        # 多头自注意力机制
        self.qkv_transform = qkv_transform(in_planes, out_planes*2, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn_qkv = nn.BatchNorm1d(out_planes*2)
        self.bn_similarity = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes*1)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride)
        self.reset_parameters()
    
    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3) # N, C, H, W -> N, H, C, W
        else:
            x = x.permute(0, 3, 1, 2) # N, C, H, W -> N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N*W, C, H) # 使x在内存中连续，并reshape为(N*W, C, H)

        # 多头自注意力机制
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N*W, self.groups, self.group_planes*2, H), [self.group_planes//2, self.group_planes//2, self.group_planes], dim=2)
        # 计算相对位置编码
        qk = torch.einsum('bgci,bgcj->bgij', q, k)
        stacked_similarity = self.bn_similarity(qk).view(N*W, 1, self.groups, H, H).sum(dim=1).contiguous()
        similarity = F.softmax(stacked_similarity, dim=-1)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sv = sv.reshape(N*W, self.out_planes*1, H).contiguous()
        output = self.bn_output(sv).view(N, W, self.out_planes, 1, H).sum(dim=-2).contiguous()
        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)
        if self.stride > 1:
            output = self.pooling(output)
        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))


class MedicalTransformer(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=256):
        super(MedicalTransformer, self).__init__()
        layers = [1, 2, 4, 2, 1] # 每个layer的AxialBlock数
        self.global_sample = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(AxialBlock_global, 8, 32, mid_planes=16, groups=8, layers=layers[0], kernel_size=img_size//2, stride=1)
        self.layer2 = self._make_layer(AxialBlock_global, 32, 64, mid_planes=32, groups=8, layers=layers[1], kernel_size=img_size//2, stride=2)
        self.decoder1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.partial_sample = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1_p = self._make_layer(AxialBlock_partial, 32, 32, mid_planes=16, groups=8, layers=layers[0], kernel_size=img_size//2, stride=1)
        self.layer2_p = self._make_layer(AxialBlock_partial, 32, 64, mid_planes=32, groups=8, layers=layers[1], kernel_size=img_size//2, stride=2)
        self.layer3_p = self._make_layer(AxialBlock_partial, 64, 128, mid_planes=64, groups=8, layers=layers[2], kernel_size=img_size//4, stride=2)
        self.layer4_p = self._make_layer(AxialBlock_partial, 128, 256, mid_planes=128, groups=8, layers=layers[3], kernel_size=img_size//8, stride=2)
        self.layer5_p = self._make_layer(AxialBlock_partial, 256, 512, mid_planes=256, groups=8, layers=layers[4], kernel_size=img_size//16, stride=2)
        self.decoder1_p = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder2_p = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder3_p = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder4_p = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder5_p = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.decoderf = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.adjust = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, in_planes, out_planes, mid_planes=64, groups=8, layers=1, kernel_size=3, stride=1):
        block_layers = []
        if layers > 1:
            block_layers.append(block(in_planes, in_planes, mid_planes=mid_planes, groups=groups, kernel_size=kernel_size, stride=1))
            for _ in range(1, layers-1):
                block_layers.append(block(in_planes, in_planes, mid_planes=mid_planes, groups=groups, kernel_size=kernel_size, stride=1))
            block_layers.append(block(in_planes, out_planes, mid_planes=mid_planes, groups=groups, kernel_size=kernel_size, stride=stride))
        else:
            block_layers.append(block(in_planes, out_planes, mid_planes=mid_planes, groups=groups, kernel_size=kernel_size, stride=stride))
        return nn.Sequential(*block_layers)

    def forward(self, x):
        xin = x.clone()
        x = self.global_sample(xin)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x = F.relu(F.interpolate(self.decoder1(x2), scale_factor=(2, 2), mode='bilinear'))
        x = torch.add(x, x1)
        x = F.relu(F.interpolate(self.decoder2(x), scale_factor=(2, 2), mode='bilinear'))
        x_loc = x.clone()
        for i in range(0, 4):
            for j in range(0, 4):
                x_p = xin[:, :, 32*i:32*(i+1), 32*j:32*(j+1)]
                x_p = self.partial_sample(x_p)
                x1_p = self.layer1_p(x_p)
                x2_p = self.layer2_p(x1_p)
                x3_p = self.layer3_p(x2_p)
                x4_p = self.layer4_p(x3_p)
                x5_p = self.layer5_p(x4_p)
                x_p = F.relu(F.interpolate(self.decoder1_p(x5_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = torch.add(x_p, x4_p)
                x_p = F.relu(F.interpolate(self.decoder2_p(x_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = torch.add(x_p, x3_p)
                x_p = F.relu(F.interpolate(self.decoder3_p(x_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = torch.add(x_p, x2_p)
                x_p = F.relu(F.interpolate(self.decoder4_p(x_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = torch.add(x_p, x1_p)
                x_p = F.relu(self.decoder5_p(x_p))
                x_loc[:, :, 32*i:32*(i+1), 32*j:32*(j+1)] = x_p
        x = torch.add(x, x_loc)
        x = F.relu(self.decoderf(x))
        x = self.adjust(x)
        return x


if __name__ == '__main__':
    model = MedicalTransformer(in_channels=3, num_classes=1, img_size=256)
    x = torch.randn(10, 3, 256, 256)
    y = model(x)
    print(y.shape)