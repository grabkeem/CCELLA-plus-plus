
from collections import OrderedDict
from typing import Optional
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
import torch
from torch import nn
from collections.abc import Sequence

from .diffusion_model_unet_maisi_prostate import DiffusionModelUNetMaisi



class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))


class PerceiverAttentionBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        normed_latents = self.ln_1(latents, timestep_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
        )
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        return latents

class PerceiverResamplerCascade(nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        output_dim=None,
        input_dim=None,
        time_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        self.time_aware_linear = nn.Linear(
            time_embedding_dim or width, width, bias=True
        )

        if self.input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width, heads, time_embedding_dim=time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        self.cascade_blocks = nn.Sequential(
            *[
                ClassAdaptor()
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor = None):
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
        latents = learnable_latents + self.time_aware_linear(
            torch.nn.functional.silu(timestep_embedding)
        )
        if self.input_dim is not None:
            x = self.proj_in(x)
        class_outs = []
        for i in range(len(self.perceiver_blocks)):
            p_block = self.perceiver_blocks[i]
            c_block = self.cascade_blocks[i]
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)
            class_outs.append(c_block(latents))

        class_out = torch.hstack(class_outs)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents, class_out

class CCELLA(nn.Module):
    def __init__(
        self,
        time_channel=320,
        time_embed_dim=768,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        width=768,
        layers=6,
        heads=8,
        num_latents=64,
        input_dim=2048,
    ):
        super().__init__()

        self.position = Timesteps(
            time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
        )

        self.connector = PerceiverResamplerCascade(
            width=width,
            layers=layers,
            heads=heads,
            num_latents=num_latents,
            input_dim=input_dim,
            time_embedding_dim=time_embed_dim,
        )

    def forward(self, text_encode_features, timesteps):
        device = text_encode_features.device
        dtype = text_encode_features.dtype

        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        ori_time_feature = (
            ori_time_feature.unsqueeze(dim=1)
            if ori_time_feature.ndim == 2
            else ori_time_feature
        )
        ori_time_feature = ori_time_feature.expand(len(text_encode_features), -1, -1)
        time_embedding = self.time_embedding(ori_time_feature)

        encoder_hidden_states, class_hidden_states = self.connector(
            text_encode_features, timestep_embedding=time_embedding
        )

        return encoder_hidden_states, class_hidden_states

class ClassAdaptor(nn.Module):
    def __init__(self, in_channels: int=256*768, hidden_channels: int=256, num_classes: int=2):
        super().__init__()

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x.reshape(x.shape[0], -1))
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CCELLA_LDM(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
            num_channels: Sequence[int] = (32, 64, 64, 64),
            attention_levels: Sequence[bool] = (False, False, True, True),
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            resblock_updown: bool = False,
            num_head_channels: int | Sequence[int] = 8,
            with_conditioning: bool = False,
            transformer_num_layers: int = 1,
            cross_attention_dim: int | None = None,
            num_class_embeds: int | None = None,
            upcast_attention: bool = False,
            include_fc: bool = False,
            use_combined_linear: bool = False,
            use_flash_attention: bool = False,
            dropout_cattn: float = 0.0,
            include_pirads_input: bool = False,
            include_spacing_input: bool = False,

            time_channel=320,
            time_embed_dim=768,
            act_fn: str = "silu",
            out_dim: Optional[int] = None,
            width=768,
            layers=6,
            heads=8,
            num_latents=256,
            input_dim=2048,

            num_classes: int = 2,
            num_class_hidden: int = 32,
            ):
        super().__init__()
        self.unet = DiffusionModelUNetMaisi(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            attention_levels=attention_levels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            with_conditioning=with_conditioning,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
            include_pirads_input=include_pirads_input,
            include_spacing_input=include_spacing_input,
        )
        self.ella = CCELLA(
            time_channel=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
            width=width,
            layers=layers,
            heads=heads,
            num_latents=num_latents,
            input_dim=input_dim,
        )
        self.classifier = nn.Linear(layers*num_classes, num_classes)

    def forward(self, x, timesteps, spacing_tensor, text_encoding):
        context, class_hidden = self.ella(text_encoding, timesteps)
        class_pred_pre = self.classifier(class_hidden)
        class_pred = torch.softmax(class_pred_pre, dim=-1)*1e2
        x= self.unet(
                    x=x,
                    context=context,
                    timesteps=timesteps,
                    pirads=class_pred,
                    spacing_tensor=spacing_tensor,
                )
        return x, class_pred_pre