from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck
from .autoencoder import get_encoder_decoder
import torch

class AutoEncoderCompAI(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y
            x ──►─┤g_a├──►─┐
                  └───┘    │
                           ▼
                         ┌─┴─┐
                         │ Q │
                         └─┬─┘
                           │
                     y_hat ▼
                           │
                           ·
                        EB :
                           ·
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
    """
    def __init__(self, input_dim = 96, hierarchical_structure = True):
        super().__init__()

        self.g_a, self.g_s, self.latent_dim, self.pad_input = get_encoder_decoder(input_dim, hierarchical_structure)

        self.entropy_bottleneck = EntropyBottleneck(self.latent_dim)

    def forward(self, x):
        B,N,C = x.shape
        remainder = None
        if self.pad_input and N % 8 != 0:
            remainder = N % 8
            pad_tokens = torch.ones(B, 8-remainder, C).to(x.device)
            x = torch.cat([x, pad_tokens], dim=1)

        y = self.g_a(x)
        y = y.permute(0,2,1) # B,C,N

        y_hat, y_likelihoods = self.entropy_bottleneck(y)

        y_hat = y_hat.permute(0,2,1)
        
        x_hat = self.g_s(y_hat)

        if self.pad_input and remainder is not None:
            x_hat = x_hat[:,:-(8-remainder),:]

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }
    
    def compress(self, x):
        B,N,C = x.shape
        remainder = None
        if self.pad_input and N % 8 != 0:
            remainder = N % 8
            pad_tokens = torch.ones(B, 8-remainder, C).to(x.device)
            x = torch.cat([x, pad_tokens], dim=1)

        y = self.g_a(x)
        y = y.permute(0,2,1)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": [y.size()[-1]], "remainder":remainder}
    
    def decompress(self, strings, shape, remainder):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        y_hat = y_hat.permute(0,2,1)
        x_hat = self.g_s(y_hat)

        if self.pad_input and remainder is not None:
            x_hat = x_hat[:,:-(8-remainder),:]

        return {"x_hat": x_hat}

if __name__ == '__main__':
    device = 'cuda'
    x = torch.rand((8,190,768)).to(device) # B,N,C

    model = AutoEncoderCompAI(input_dim=768, hierarchical_structure=True).to(device)
    model.update()
    out_enc = model.compress(x)
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], out_enc["remainder"])

    print(out_dec["x_hat"].shape)