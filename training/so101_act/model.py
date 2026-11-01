"""Compact goal-conditioned ACT: spatial CNN + transformer + action CVAE.

Independent implementation of the ACT design (Zhao et al., RSS 2023), not an
upstream/LeRobot checkpoint-compatible replica. No pretrained weights downloaded.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from action_generator.goal import GOAL_DIM


@dataclass
class ACTConfig:
    image_size: tuple[int, int] = (120, 160)
    chunk_size: int = 16
    hidden_dim: int = 128
    latent_dim: int = 16
    nheads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    posterior_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = .1
    residual_actions: bool = True
    include_velocity: bool = False
    include_goal_error: bool = False

    def __post_init__(self):
        if type(self.residual_actions) is not bool:
            raise ValueError('residual_actions must be boolean')
        if type(self.include_velocity) is not bool:
            raise ValueError('include_velocity must be boolean')
        if type(self.include_goal_error) is not bool:
            raise ValueError('include_goal_error must be boolean')
        self.image_size = tuple(self.image_size)
        values = (self.chunk_size, self.hidden_dim, self.latent_dim, self.nheads,
                  self.encoder_layers, self.decoder_layers, self.posterior_layers, self.feedforward_dim)
        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError('ACT dimensions/layer counts must be positive integers')
        if self.hidden_dim % self.nheads or not 0 <= self.dropout < 1:
            raise ValueError('invalid attention dimension or dropout')
        if len(self.image_size) != 2 or any(type(v) is not int or v < 32 for v in self.image_size):
            raise ValueError('image_size must be [height, width], both >=32')

    def to_dict(self):
        return asdict(self)


class GoalConditionedACT(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = c = config
        d = c.hidden_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, d, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((6, 8)))
        self.image_pos = nn.Parameter(torch.randn(1, 48, d)*.02)
        self.condition_pos = nn.Parameter(torch.randn(1, 3, d)*.02)
        self.state_proj = nn.Linear(6+6*c.include_velocity+3*c.include_goal_error, d)
        self.goal_proj = nn.Linear(GOAL_DIM, d)
        self.action_proj = nn.Linear(6, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.posterior_pos = nn.Parameter(torch.randn(1, c.chunk_size+3, d)*.02)
        def encoder(layers):
            layer = nn.TransformerEncoderLayer(d, c.nheads, c.feedforward_dim,
                                               c.dropout, batch_first=True)
            return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.posterior = encoder(c.posterior_layers)
        self.latent_params = nn.Linear(d, 2*c.latent_dim)
        self.latent_proj = nn.Linear(c.latent_dim, d)
        self.encoder = encoder(c.encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d, c.nheads, c.feedforward_dim, c.dropout, batch_first=True), c.decoder_layers)
        self.queries = nn.Parameter(torch.randn(1, c.chunk_size, d)*.02)
        self.action_head = nn.Linear(d, 6)
        # Predict corrections around measured q rather than reconstructing its
        # entire absolute value through a visual transformer at every tick.
        self.register_buffer('state_to_action_scale', torch.ones(6))
        self.register_buffer('state_to_action_bias', torch.zeros(6))
        nn.init.normal_(self.action_head.weight, std=.001)
        nn.init.zeros_(self.action_head.bias)

    def set_normalization(self, stats):
        state, action = stats['states'], stats['actions']
        with torch.no_grad():
            self.state_to_action_scale.copy_(torch.tensor(state['std'])/torch.tensor(action['std']))
            self.state_to_action_bias.copy_((torch.tensor(state['mean'])-torch.tensor(action['mean']))/torch.tensor(action['std']))

    def forward(self, image, state, goal, actions=None, padding=None):
        batch = state.shape[0]
        state_token = self.state_proj(state).unsqueeze(1)
        goal_token = self.goal_proj(goal).unsqueeze(1)
        mu = logvar = None
        if actions is not None:
            if padding is None or padding.shape != actions.shape[:2]:
                raise ValueError('posterior requires a [B,K] padding mask')
            # Mask values too, so arbitrary padded contents cannot contaminate attention.
            action_tokens = self.action_proj(actions.masked_fill(padding.unsqueeze(-1), 0.))
            tokens = torch.cat((self.cls.expand(batch, -1, -1), state_token, goal_token, action_tokens), 1)
            mask = torch.cat((torch.zeros(batch, 3, dtype=torch.bool, device=state.device), padding), 1)
            summary = self.posterior(tokens+self.posterior_pos, src_key_padding_mask=mask)[:, 0]
            mu, logvar = self.latent_params(summary).chunk(2, -1)
            logvar = logvar.clamp(-10., 10.)
            z = mu + torch.randn_like(mu)*torch.exp(.5*logvar)
        else:
            z = state.new_zeros((batch, self.config.latent_dim))
        visual = self.backbone(image).flatten(2).transpose(1, 2)+self.image_pos
        condition = torch.cat((state_token, goal_token, self.latent_proj(z).unsqueeze(1)), 1)
        memory = self.encoder(torch.cat((condition+self.condition_pos, visual), 1))
        decoded = self.decoder(self.queries.expand(batch, -1, -1), memory)
        base = state[:, :6]*self.state_to_action_scale+self.state_to_action_bias
        prediction = self.action_head(decoded)
        return (base.unsqueeze(1)+prediction if self.config.residual_actions else prediction), mu, logvar


def act_loss(prediction, actions, padding, mu=None, logvar=None, kl_weight=1.):
    valid = (~padding).unsqueeze(-1)
    if not valid.any():
        raise ValueError('action chunk has no valid labels')
    reconstruction = ((prediction-actions).abs().masked_fill(~valid, 0.)).sum()/(valid.sum()*actions.shape[-1])
    kl = prediction.new_zeros(()) if mu is None else -.5*(1+logvar-mu.square()-logvar.exp()).sum(-1).mean()
    return reconstruction+kl_weight*kl, reconstruction, kl
