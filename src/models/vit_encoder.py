from .utils import FeedForward, Attention
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class PatchMerging(nn.Module):

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm, down_factor = 2):
        super().__init__()
        self.dim = dim
        self.down_factor = down_factor
        self.reduction = nn.Linear(down_factor * dim, out_dim, bias=False)
        self.norm = norm_layer(down_factor * dim)
        

    def forward(self, x):
        """
        x: B, N, C
        """
        B, N, C = x.shape
        # assert N % down_factor == 0, f"x lnght is not even."

        remainder = N % self.down_factor
        if remainder != 0:
            pad_tokens = x[:, -1:, :].repeat(1, self.down_factor - remainder, 1)  # (B, self.down_factor - remainder, C)
            x = torch.cat([x, pad_tokens], dim=1)
            N = x.shape[1]  

        num_patch = N // self.down_factor
        x = x.view(B, num_patch, self.down_factor, C)
        x = x.view(B, num_patch, self.down_factor * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PreNorm(nn.Module):
    def __init__(self, norm_layer, dim, fn):
        super().__init__()
        self.norm = norm_layer(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class ViTBlock(nn.Module):
    def __init__(self, 
                 dim: int,
                 num_heads: int,
                 mlp_ratio = 4,
                 dropout = 0.,
                 norm_layer = nn.LayerNorm):
        super().__init__()

        assert dim%num_heads == 0, f'trying to divide vector of dimension: {dim} in {num_heads} heads..'

        self.att = PreNorm(
            norm_layer=norm_layer, dim=dim, 
            fn = Attention(dim, num_heads, dim_head = dim // num_heads, dropout = dropout)
        )
        self.ffn = PreNorm(
            norm_layer=norm_layer, dim=dim,
            fn = FeedForward(dim, hidden_dim = mlp_ratio*dim, dropout = dropout)
        )

    def forward(self, x):
        x = self.att(x) + x
        x = self.ffn(x) + x
        return x


class BasicBlock(nn.Module):
    # implements sigle stage model
    def __init__(self, 
                 dim:int, 
                 depth:int,
                 out_dim:int, 
                 num_heads:int, 
                 mlp_ratio = 4,
                 dropout = 0.,
                 norm_layer = nn.LayerNorm,
                 downsample = PatchMerging,
                 down_factor = 2):
        super().__init__()


        self.blocks = nn.ModuleList([
            ViTBlock(
                dim = dim,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                dropout = dropout,
                norm_layer = norm_layer
            ) for _ in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(dim = dim, out_dim=out_dim, norm_layer=norm_layer, down_factor=down_factor)
        else:
            self.downsample = None
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)
        
        return x


class HierarchicalEncoderViT(nn.Module):
    def __init__(self,
                 dims =     [96,96*2,96*3,96*4],
                 depths =   [2, 2,   6,   2],
                 heads =    [3, 6,   12,  24],
                 mlp_ratio = 4,
                 dropout = 0.,
                 norm_layer = nn.LayerNorm,
                 downsample = PatchMerging,
                 down_factors = [2,2,2,None]):

        super().__init__()

        assert len(dims) == len(depths) and len(depths) == len(heads)

        dims.append(-1) # ignore it

            
        self.layers = nn.ModuleList([
            BasicBlock(
                dim=dims[i],
                depth=depths[i],
                out_dim=dims[i+1],
                num_heads=heads[i],
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                norm_layer=norm_layer,
                downsample=downsample if (i < len(depths) -1) else None,
                down_factor = down_factors[i]
            ) for i in range(len(depths))
        ])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    


if __name__ == '__main__':

    device = "cuda"

    x = torch.rand((8,512,96)).to(device)

    encoder = HierarchicalEncoderViT(
        dims = [96,32,96,100],
        heads=[2,2,2,4],
        down_factors=[2,2,2,None]
    ).to(device)

    print(encoder(x).shape)
    
