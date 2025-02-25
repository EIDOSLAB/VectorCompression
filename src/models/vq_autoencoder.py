from .autoencoder import AutoEncoder
from vector_quantize_pytorch import VectorQuantize
import torch


class AutoEncoderVQ(AutoEncoder):
    def __init__(self, input_dim=96, hierarchical_structure=True, codebook_size = 1024):
        super().__init__(input_dim, hierarchical_structure)

        self.vq = VectorQuantize(dim = self.latent_dim, codebook_size=codebook_size)


    def forward(self, x):
        
        B,N,C = x.shape
        remainder = None
        if self.pad_input and N % 8 != 0:
            remainder = N % 8
            pad_tokens = torch.ones(B, 8-remainder, C).to(x.device)
            x = torch.cat([x, pad_tokens], dim=1)

        y = self.encoder(x)

        y_hat, idxs, commit_loss = self.vq(y)
        # print(y_hat.shape)
        # print(idxs.shape)


        x_hat = self.decoder(y_hat)

        if self.pad_input and remainder is not None:
            x_hat = x_hat[:,:-(8-remainder),:]

        return x_hat, commit_loss



if __name__ == '__main__':
    device = 'cuda'
    x = torch.rand((8,192,768)).to(device) # B,N,C

    ae = AutoEncoderVQ(input_dim=768, hierarchical_structure=False).to(device)

    x_hat, loss = ae(x)