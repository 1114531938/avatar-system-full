import unittest

import torch

from scene.gaussian_model import GaussianModel
from scene.micro_expression import GiGAStyleOffsetModule


class BoundGaussianFixture(GaussianModel):
    def __init__(self, sh_degree=0):
        super().__init__(sh_degree=sh_degree)
        self.binding = torch.tensor([0, 1])
        self.face_center = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        self.face_scaling = torch.ones(2, 1)
        self.face_orien_mat = torch.eye(3).repeat(2, 1, 1)
        self.face_orien_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        self._xyz = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self._scaling = torch.zeros(2, 3)
        self._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        self._opacity = torch.zeros(2, 1)
        self._features_dc = torch.zeros(2, 1, 3)
        self._features_rest = torch.zeros(2, (sh_degree + 1) ** 2 - 1, 3)
        self.offsets = None

    def get_attribute_offset(self, name):
        if self.offsets is None:
            return None
        return self.offsets.get(name)


class GiGAStyleOffsetModuleTest(unittest.TestCase):
    def test_zero_initialized_offsets_do_not_change_bound_attributes(self):
        model = BoundGaussianFixture()
        reference = {
            "xyz": model.get_xyz.clone(),
            "scaling": model.get_scaling.clone(),
            "opacity": model.get_opacity.clone(),
            "rotation": model.get_rotation.clone(),
            "features": model.get_features.clone(),
        }
        module = GiGAStyleOffsetModule(anchor_count=2, topk=1)
        model.offsets = module(
            torch.randn(2, 13),
            model.binding,
            model._xyz,
            model._scaling,
            model._opacity,
            model._rotation,
            model._features_dc.squeeze(1),
        )

        torch.testing.assert_close(model.get_xyz, reference["xyz"])
        torch.testing.assert_close(model.get_scaling, reference["scaling"])
        torch.testing.assert_close(model.get_opacity, reference["opacity"])
        torch.testing.assert_close(model.get_rotation, reference["rotation"])
        torch.testing.assert_close(model.get_features, reference["features"])

    def test_property_hooks_apply_residuals_when_offsets_are_nonzero(self):
        model = BoundGaussianFixture()
        baseline_xyz = model.get_xyz.clone()
        baseline_opacity = model.get_opacity.clone()
        model.offsets = {
            "xyz": torch.full_like(model._xyz, 0.25),
            "scaling": torch.zeros_like(model._scaling),
            "opacity": torch.ones_like(model._opacity),
            "rotation": torch.zeros_like(model._rotation),
            "features_dc": torch.zeros_like(model._features_dc),
        }

        self.assertFalse(torch.equal(model.get_xyz, baseline_xyz))
        self.assertTrue(torch.all(model.get_opacity > baseline_opacity))

    def test_v2_offsets_are_bounded_to_small_residuals(self):
        module = GiGAStyleOffsetModule(anchor_count=2, topk=1)
        with torch.no_grad():
            module.offset_mlp[-1].bias.fill_(100.0)
        offsets = module(
            torch.randn(2, 13),
            torch.tensor([0, 1]),
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.zeros(2, 1),
            torch.zeros(2, 4),
            torch.zeros(2, 3),
        )

        self.assertLessEqual(offsets["xyz"].abs().max().item(), 0.010001)
        self.assertLessEqual(offsets["scaling"].abs().max().item(), 0.050001)
        self.assertLessEqual(offsets["opacity"].abs().max().item(), 0.100001)
        self.assertLessEqual(offsets["rotation"].abs().max().item(), 0.050001)
        self.assertLessEqual(offsets["features_dc"].abs().max().item(), 0.030001)

    def test_separate_attribute_heads_are_zero_initialized(self):
        model = BoundGaussianFixture()
        reference = {
            "xyz": model.get_xyz.clone(),
            "scaling": model.get_scaling.clone(),
            "opacity": model.get_opacity.clone(),
            "rotation": model.get_rotation.clone(),
            "features": model.get_features.clone(),
        }
        module = GiGAStyleOffsetModule(anchor_count=2, topk=1, head_mode="separate")
        model.offsets = module(
            torch.randn(2, 13),
            model.binding,
            model._xyz,
            model._scaling,
            model._opacity,
            model._rotation,
            model._features_dc.squeeze(1),
            model._features_rest,
        )

        torch.testing.assert_close(model.get_xyz, reference["xyz"])
        torch.testing.assert_close(model.get_scaling, reference["scaling"])
        torch.testing.assert_close(model.get_opacity, reference["opacity"])
        torch.testing.assert_close(model.get_rotation, reference["rotation"])
        torch.testing.assert_close(model.get_features, reference["features"])

    def test_separate_attribute_heads_can_change_one_attribute(self):
        module = GiGAStyleOffsetModule(anchor_count=2, topk=1, head_mode="separate")
        with torch.no_grad():
            module.attribute_mlps["opacity"][-1].bias.fill_(100.0)
        offsets = module(
            torch.randn(2, 13),
            torch.tensor([0, 1]),
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.zeros(2, 1),
            torch.zeros(2, 4),
            torch.zeros(2, 3),
            torch.zeros(2, 0, 3),
        )

        self.assertEqual(offsets["xyz"].abs().max().item(), 0.0)
        self.assertLessEqual(offsets["opacity"].abs().max().item(), 0.100001)
        self.assertGreater(offsets["opacity"].abs().max().item(), 0.09)

    def test_sh_appearance_mode_offsets_rest_features(self):
        model = BoundGaussianFixture(sh_degree=1)
        baseline_features = model.get_features.clone()
        module = GiGAStyleOffsetModule(
            anchor_count=2,
            topk=1,
            head_mode="separate",
            appearance_mode="sh",
            sh_degree=1,
        )
        with torch.no_grad():
            module.attribute_mlps["features_rest"][-1].bias.fill_(100.0)
        model.offsets = module(
            torch.randn(2, 13),
            model.binding,
            model._xyz,
            model._scaling,
            model._opacity,
            model._rotation,
            model._features_dc.squeeze(1),
            model._features_rest,
        )

        self.assertFalse(torch.equal(model.get_features, baseline_features))
        self.assertLessEqual(model.offsets["features_rest"].abs().max().item(), 0.030001)

    def test_legacy_unbounded_attributes_remain_loadable(self):
        module = GiGAStyleOffsetModule(
            anchor_count=2,
            topk=1,
            scaling_scale=None,
            opacity_scale=None,
            rotation_scale=None,
            appearance_scale=None,
        )
        with torch.no_grad():
            module.offset_mlp[-1].bias.fill_(1.0)
        offsets = module(
            torch.randn(2, 13),
            torch.tensor([0, 1]),
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.zeros(2, 1),
            torch.zeros(2, 4),
            torch.zeros(2, 3),
        )
        self.assertGreater(offsets["scaling"].abs().max().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
