
import torch

from torch import nn
from torch.nn import functional as F
from vits import attentions
from vits import commons
from vits import modules
from vits.utils import f0_to_coarse
from vits.modules_grl import SpeakerClassifier


class TextEncoder(nn.Module):
    def __init__(self,
                 vec_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(vec_channels, hidden_channels, kernel_size=5, padding=2)
        self.enc = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, f0):
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = x + f0
        x = self.enc(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask, x


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, pit, g=None, reverse=False):
        if not reverse:
            total_logdet = 0
            for flow in self.flows:
                x, log_det = flow(x, x_mask, pit, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet
        else:
            total_logdet = 0
            for flow in reversed(self.flows):
                x, log_det = flow(x, x_mask, pit, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()


class SynthesizerTrn(nn.Module):
    def __init__(self, spec_channels, segment_size, hp):
        super().__init__()
        self.segment_size = segment_size
        self.emb_p = nn.Embedding(256, hp.vits.hidden_channels)
        self.enc_p = TextEncoder(
            hp.vits.vec_dim,
            spec_channels,
            hp.vits.hidden_channels,
            hp.vits.filter_channels,
            2,
            6,
            3,
            0.1,
        )
        self.speaker_classifier = SpeakerClassifier(
            hp.vits.hidden_channels,
            hp.vits.spk_dim,
        )
        self.flow = ResidualCouplingBlock(
            spec_channels,
            hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=hp.vits.spk_dim
        )

    def forward(self, vec, pit, spec, spk, vec_l, spec_l):
        vec = vec + torch.randn_like(vec) * 1  # Perturbation

        pit = self.emb_p(f0_to_coarse(pit)).transpose(1, 2)
        z_p, m_p, logs_p, vec_mask, x = self.enc_p(vec, vec_l, pit)

        # SNAC to flow
        z_f, logdet_f = self.flow(spec, vec_mask, pit, g=spk)
        z_r, logdet_r = self.flow(z_p, vec_mask, pit, g=spk, reverse=True)
        # speaker
        spk_preds = self.speaker_classifier(x)
        return z_r, vec_mask, (z_f, logdet_f, m_p, logs_p), spk_preds

    def infer(self, vec, pit, spk, vec_l):
        pit = self.emb_p(f0_to_coarse(pit)).transpose(1, 2)
        z_p, m_p, logs_p, vec_mask, x = self.enc_p(vec, vec_l, pit)
        z, _ = self.flow(z_p, vec_mask, pit, g=spk, reverse=True)
        return z
