import math

import torch
from torch import nn


class GiGAStyleOffsetModule(nn.Module):
    """Sparse global mesh context followed by per-Gaussian residual offsets."""

    TRIANGLE_FEATURE_DIM = 13

    def __init__(
        self,
        hidden_dim=64,
        embed_dim=32,
        anchor_count=256,
        topk=8,
        head_mode="shared",
        appearance_mode="dc",
        sh_degree=3,
        position_scale=0.01,
        scaling_scale=0.05,
        opacity_scale=0.1,
        rotation_scale=0.05,
        appearance_scale=0.03,
    ):
        super().__init__()
        self.anchor_count = anchor_count
        self.topk = topk
        self.head_mode = head_mode
        self.appearance_mode = appearance_mode
        self.sh_degree = sh_degree
        self.position_scale = position_scale
        self.scaling_scale = scaling_scale
        self.opacity_scale = opacity_scale
        self.rotation_scale = rotation_scale
        self.appearance_scale = appearance_scale

        if self.head_mode not in {"shared", "separate"}:
            raise ValueError(f"Unsupported micro-expression head mode: {self.head_mode}")
        if self.appearance_mode not in {"dc", "sh"}:
            raise ValueError(f"Unsupported micro-expression appearance mode: {self.appearance_mode}")
        if self.head_mode == "shared" and self.appearance_mode != "dc":
            raise ValueError("appearance_mode='sh' requires head_mode='separate'")

        self.query = nn.Linear(self.TRIANGLE_FEATURE_DIM, embed_dim)
        self.key = nn.Linear(self.TRIANGLE_FEATURE_DIM, embed_dim)

        gaussian_feature_dim = 3 + 3 + 1 + 4 + 3
        self.features_rest_dim = ((self.sh_degree + 1) ** 2 - 1) * 3
        if self.head_mode == "shared":
            self.offset_mlp = self._make_mlp(
                self.TRIANGLE_FEATURE_DIM * 2 + gaussian_feature_dim,
                gaussian_feature_dim,
                hidden_dim,
            )
        else:
            self.attribute_mlps = nn.ModuleDict({
                "xyz": self._make_mlp(self.TRIANGLE_FEATURE_DIM * 2 + 3, 3, hidden_dim),
                "scaling": self._make_mlp(self.TRIANGLE_FEATURE_DIM * 2 + 3, 3, hidden_dim),
                "opacity": self._make_mlp(self.TRIANGLE_FEATURE_DIM * 2 + 1, 1, hidden_dim),
                "rotation": self._make_mlp(self.TRIANGLE_FEATURE_DIM * 2 + 4, 4, hidden_dim),
                "features_dc": self._make_mlp(self.TRIANGLE_FEATURE_DIM * 2 + 3, 3, hidden_dim),
            })
            if self.appearance_mode == "sh" and self.features_rest_dim > 0:
                self.attribute_mlps["features_rest"] = self._make_mlp(
                    self.TRIANGLE_FEATURE_DIM * 2 + self.features_rest_dim,
                    self.features_rest_dim,
                    hidden_dim,
                )

    @staticmethod
    def _make_mlp(in_dim, out_dim, hidden_dim):
        mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(mlp[-1].weight)
        nn.init.zeros_(mlp[-1].bias)
        return mlp

    @staticmethod
    def _bounded_offset(offset, scale):
        if scale is None:
            return offset
        return torch.tanh(offset) * scale

    def _global_triangle_features(self, triangle_features):
        num_faces = triangle_features.shape[0]
        num_anchors = min(self.anchor_count, num_faces)
        anchor_indices = torch.linspace(
            0,
            num_faces - 1,
            steps=num_anchors,
            device=triangle_features.device,
        ).long()
        anchor_features = triangle_features[anchor_indices]

        queries = self.query(triangle_features)
        keys = self.key(anchor_features)
        logits = queries @ keys.transpose(0, 1) / math.sqrt(keys.shape[-1])

        k = min(self.topk, num_anchors)
        values, indices = torch.topk(logits, k=k, dim=-1)
        weights = torch.softmax(values, dim=-1)
        selected_features = anchor_features[indices]
        return (weights[..., None] * selected_features).sum(dim=1)

    def forward(
        self,
        triangle_features,
        binding,
        xyz,
        scaling,
        opacity,
        rotation,
        features_dc,
        features_rest=None,
    ):
        global_features = self._global_triangle_features(triangle_features)
        local_features = triangle_features[binding]
        global_features = global_features[binding]
        context = torch.cat((local_features, global_features), dim=-1)

        if self.head_mode == "separate":
            offsets = {
                "xyz": torch.tanh(self.attribute_mlps["xyz"](torch.cat((context, xyz), dim=-1))) * self.position_scale,
                "scaling": self._bounded_offset(
                    self.attribute_mlps["scaling"](torch.cat((context, scaling), dim=-1)),
                    self.scaling_scale,
                ),
                "opacity": self._bounded_offset(
                    self.attribute_mlps["opacity"](torch.cat((context, opacity), dim=-1)),
                    self.opacity_scale,
                ),
                "rotation": self._bounded_offset(
                    self.attribute_mlps["rotation"](torch.cat((context, rotation), dim=-1)),
                    self.rotation_scale,
                ),
                "features_dc": self._bounded_offset(
                    self.attribute_mlps["features_dc"](torch.cat((context, features_dc), dim=-1)),
                    self.appearance_scale,
                )[:, None, :],
            }
            if "features_rest" in self.attribute_mlps:
                if features_rest is None:
                    raise ValueError("features_rest is required when appearance_mode='sh'")
                features_rest_flat = features_rest.reshape(features_rest.shape[0], -1)
                features_rest_offset = self._bounded_offset(
                    self.attribute_mlps["features_rest"](torch.cat((context, features_rest_flat), dim=-1)),
                    self.appearance_scale,
                )
                offsets["features_rest"] = features_rest_offset.reshape_as(features_rest)
            return offsets

        gaussian_features = torch.cat(
            (xyz, scaling, opacity, rotation, features_dc),
            dim=-1,
        )
        offsets = self.offset_mlp(torch.cat((context, gaussian_features), dim=-1))
        xyz_offset, scaling_offset, opacity_offset, rotation_offset, appearance_offset = (
            torch.split(offsets, (3, 3, 1, 4, 3), dim=-1)
        )
        return {
            "xyz": torch.tanh(xyz_offset) * self.position_scale,
            "scaling": self._bounded_offset(scaling_offset, self.scaling_scale),
            "opacity": self._bounded_offset(opacity_offset, self.opacity_scale),
            "rotation": self._bounded_offset(rotation_offset, self.rotation_scale),
            "features_dc": self._bounded_offset(appearance_offset, self.appearance_scale)[:, None, :],
        }
