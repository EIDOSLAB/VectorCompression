import torch.nn as nn
from .vit_encoder import ViTBlock

from timm.models.layers import trunc_normal_
import torch

from einops import rearrange



class PatchSplit(nn.Module):

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm, up_factor = 2):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.reduction = nn.Linear(dim, out_dim * up_factor, bias=False)

        self.up_factor = up_factor
        self.out_dim = out_dim

    def forward(self, x):
        """
        x: B, N, C
        """

        x = self.norm(x)
        x = self.reduction(x) 
        x = rearrange(x, 'b n (c r) -> b (n r) c', c = self.out_dim, r = self.up_factor)
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
                 upsample = PatchSplit,
                 up_factor = 2):
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

        if upsample is not None:
            self.upsample = upsample(dim = dim, out_dim=out_dim, norm_layer=norm_layer, up_factor=up_factor)
        else:
            self.upsample = None
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class HierarchicalDecoderViT(nn.Module):
    def __init__(self,
                 dims =     [96*4,96*3,96*2,96],
                 depths =   [2,   6,   2,   2],
                 heads =    [24,  12,  6,   3],
                 mlp_ratio = 4,
                 dropout = 0.,
                 norm_layer = nn.LayerNorm,
                 upsample = PatchSplit,
                 up_factors = [2,2,2,None]):

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
                upsample=upsample if (i < len(depths) -1) else None,
                up_factor = up_factors[i]
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

    x = torch.rand((8,512,96*4)).to(device)

    decoder = HierarchicalDecoderViT().to(device)

    print(decoder(x).shape)