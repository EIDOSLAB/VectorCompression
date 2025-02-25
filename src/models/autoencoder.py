import torch
from .vit_decoder import HierarchicalDecoderViT, PatchSplit
from .vit_encoder import HierarchicalEncoderViT, PatchMerging
import torch.nn as nn


def get_encoder_decoder(input_dim = 96, hierarchical_structure = True):

    depths = [2,2,6,2]

    # TODO play with these hyps
    if hierarchical_structure:
        # dims = [ input_dim for _ in range(4)] # [ 96, 96, ... 96]
        dims = [ input_dim * (i+1) for i in range(4)] # [96, 96*2, .. 96*4]
        heads = [12, 24, 32, 48]
        scaling_factors=[2,2,2,-1]
        downsample = PatchMerging
        upsample = PatchSplit
        pad_input = True
    else:
        dims = [ input_dim for _ in range(4)] # [ 96, 96, ... 96]
        heads = [12, 12, 12, 12]
        scaling_factors=[1,1,1,-1]
        downsample = None
        upsample = None
        pad_input = False

    latent_dim = dims[-1]

    encoder = HierarchicalEncoderViT(
        dims = dims,
        depths = depths,
        heads = heads,
        mlp_ratio=4,
        dropout=0.,
        downsample=downsample,
        down_factors=scaling_factors
    )

    dims = dims[:-1]
    dims = dims[::-1]
    depths = depths[::-1]
    heads = heads[::-1]
    decoder = HierarchicalDecoderViT(
        dims = dims,
        depths = depths,
        heads = heads,
        mlp_ratio=4,
        dropout=0.,
        upsample=upsample,
        up_factors=scaling_factors
    )

    return encoder, decoder, latent_dim, pad_input


class AutoEncoder(nn.Module):
    def __init__(self, input_dim = 96, hierarchical_structure = True) -> None:
        super().__init__()

        self.encoder, self.decoder, self.latent_dim, self.pad_input = get_encoder_decoder(input_dim, hierarchical_structure)
        
        

    def forward(self, x):
        # x: B,N,C

        B,N,C = x.shape
        remainder = None
        if self.pad_input and N % 8 != 0:
            remainder = N % 8
            pad_tokens = torch.ones(B, 8-remainder, C).to(x.device)
            x = torch.cat([x, pad_tokens], dim=1)

        x = self.encoder(x)

        x = self.decoder(x)

        if self.pad_input and remainder is not None:
            x = x[:,:-(8-remainder),:]

        return x
    
if __name__ == '__main__':
    device = 'cuda'
    x = torch.rand((8,192,768)).to(device) # B,N,C

    ae = AutoEncoder(input_dim=768, hierarchical_structure=True).to(device)

    ae(x)



