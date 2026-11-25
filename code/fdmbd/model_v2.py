"""Prospective second candidate: identity-safe polar-endpoint RF refinement.

This is a newly specified conditional refinement decoder, not recovered S0.
The bounds prevent an untrained residual from creating an arbitrary noise floor.
"""
import torch
from model import MultibandFlow


class EndpointFlow(MultibandFlow):
    log_gain_bound = .25
    phase_bound = .25

    def endpoint(self, state, t, codec_spectrum, latent):
        fields = super().forward(state, t, codec_spectrum, latent)
        polar = self.transform.unpack(fields)
        base = self.transform.unpack(codec_spectrum)
        correction = torch.complex(self.log_gain_bound * torch.tanh(polar.real),
                                   self.phase_bound * torch.tanh(polar.imag))
        return self.transform.pack(base * correction.exp())

    def forward(self, state, t, codec_spectrum, latent):
        endpoint = self.endpoint(state, t, codec_spectrum, latent)
        return (endpoint-state)/(1-t[:,None,None,None]).clamp_min(.0001)
