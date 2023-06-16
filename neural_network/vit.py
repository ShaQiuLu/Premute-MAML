import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from collections import OrderedDict

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, i, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.i = str(i)
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.l1 = nn.Linear(dim, hidden_dim)
        # self.l1 = nn.Conv2d(dim, hidden_dim, kernel_size=dim, stride=dim)
        self.GELU = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(hidden_dim, dim)
        self.dropout1 = nn.Dropout(dropout)


    def forward(self, x, params, base):
        out = F.layer_norm(x, normalized_shape=[self.dim], weight=params[base + '.' + self.i + '.1.' + 'norm.weight'],
                           bias=params[base + '.' + self.i + '.1.' + 'norm.bias'])
        # print(out.shape)
        out = F.linear(out, weight=params[base + '.' + self.i + '.1.' + 'l1.weight'],
                       bias=params[base + '.' + self.i + '.1.' + 'l1.bias'])
        # print(out.shape)
        out = self.GELU(out)
        out = self.dropout(out)
        out = F.linear(out, weight=params[base + '.' + self.i + '.1.' + 'l2.weight'],
                       bias=params[base + '.' + self.i + '.1.' + 'l2.bias'])
        out = self.dropout1(out)

        return out

class Attention(nn.Module):
    def __init__(self, i, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.i = str(i)
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x, params, base):
        x = F.layer_norm(x, normalized_shape=[self.dim], weight=params[base + '.' + self.i + '.0.' + 'norm.weight'],
                           bias=params[base + '.' + self.i + '.0.' + 'norm.bias'])
        qkv = F.linear(x, weight=params[base+'.'+self.i+'.0.'+'to_qkv.weight'],).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        # print(out.shape)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print(out.shape)
        out = F.linear(out, weight=params[base+'.'+self.i+'.0.'+'to_out.weight'],
                       bias=params[base+'.'+self.i+'.0.'+'to_out.bias'])
        out = self.dropout1(out)

        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(i, dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(i, dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, params, base):
        for attn, ff in self.layers:
            x = attn(x, params=params, base=base) + x
            x = ff(x, params=params, base=base) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, num_classes=5, pool = 'cls',
                 channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_size = patch_size
        self.dim = dim

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embedding = torch.randn(1, num_patches + 1, dim)
        self.cls_token = torch.randn(1, 1, dim)
        if torch.cuda.is_available():
            self.pos_embedding = self.pos_embedding.cuda()
            self.cls_token = self.cls_token.cuda()
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.layernorm = nn.LayerNorm(dim)
        # self.fc = nn.Linear(dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')


    def forward(self, img, params=None, embedding = False):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # print(img.shape)
        # x = self.to_patch_embedding(img).flatten(2).transpose(1, 2)
        x = F.conv2d(img, weight=params['to_patch_embedding.weight'], bias=params['to_patch_embedding.bias'],
                     stride=(self.patch_size, self.patch_size)).flatten(2).transpose(1, 2)
        # print(x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, params, 'transformer.layers')   # 3,197,768
        # print(x.shape)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # print(x.shape)

        x = self.to_latent(x)

        output = F.layer_norm(x, normalized_shape=[self.dim], weight=params['layernorm.weight'],
                           bias=params['layernorm.bias'])
        print(output.shape)
        if embedding:
            return output
        else:
            # Apply Linear Layer
            logits = F.linear(output, weight=params['fc.weight'], bias=params['fc.bias'])
            return logits



if __name__ == '__main__':
    vit = ViT(
            image_size = 84,
            patch_size = 14,
            dim = 256,
            depth = 12,
            heads = 12,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            num_classes=10,
        )
    x = torch.rand(3, 3, 84, 84)
    pred = vit(x)
    # print(list(vit.parameters()))
    # params = OrderedDict(vit.named_parameters())
    # print(params)
    # for i in params:
    #     print(i)
    # print(params['fc.weight'])
    # img = torch.randn(5, 3, 84, 84)
    # y = torch.tensor([1, 3, 2, 0, 1])
    # pred = vit(img)
    # print(pred)





