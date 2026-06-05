# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import json
import numpy as np
import torch
# from vht.model.flame import FlameHead
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from .micro_expression import GiGAStyleOffsetModule
from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz


class FlameGaussianModel(GaussianModel):
    def __init__(
        self,
        sh_degree : int,
        disable_flame_static_offset=False,
        not_finetune_flame_params=False,
        n_shape=300,
        n_expr=100,
        enable_micro_expression=False,
        micro_expression_anchor_count=256,
        micro_expression_topk=8,
        micro_expression_hidden_dim=64,
        micro_expression_embed_dim=32,
        micro_expression_head_mode="shared",
        micro_expression_appearance_mode="dc",
        micro_expression_use_deformation=True,
        micro_expression_position_scale=0.01,
        micro_expression_scaling_scale=0.05,
        micro_expression_opacity_scale=0.1,
        micro_expression_rotation_scale=0.05,
        micro_expression_appearance_scale=0.03,
    ):
        super().__init__(sh_degree)

        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr
        self.micro_expression = None
        self._micro_expression_offsets = None
        self._micro_expression_regularization = None
        self._micro_expression_reference_features = None
        self._micro_expression_config = {
            "version": 3,
            "anchor_count": micro_expression_anchor_count,
            "topk": micro_expression_topk,
            "hidden_dim": micro_expression_hidden_dim,
            "embed_dim": micro_expression_embed_dim,
            "head_mode": micro_expression_head_mode,
            "appearance_mode": micro_expression_appearance_mode,
            "sh_degree": sh_degree,
            "use_deformation_features": micro_expression_use_deformation,
            "position_scale": micro_expression_position_scale,
            "scaling_scale": micro_expression_scaling_scale,
            "opacity_scale": micro_expression_opacity_scale,
            "rotation_scale": micro_expression_rotation_scale,
            "appearance_scale": micro_expression_appearance_scale,
        }
        if enable_micro_expression:
            self.enable_micro_expression()

        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
        ).cuda()
        self.flame_param = None
        self.flame_param_orig = None

        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()

    def enable_micro_expression(self, config=None, force=False):
        if config is not None:
            if "use_deformation_features" not in config:
                self._micro_expression_config.update({
                    "version": 1,
                    "head_mode": "shared",
                    "appearance_mode": "dc",
                    "use_deformation_features": False,
                    "scaling_scale": None,
                    "opacity_scale": None,
                    "rotation_scale": None,
                    "appearance_scale": None,
                })
            self._micro_expression_config.update(config)
        self._micro_expression_config.setdefault("head_mode", "shared")
        self._micro_expression_config.setdefault("appearance_mode", "dc")
        self._micro_expression_config.setdefault("sh_degree", self.max_sh_degree)
        if force:
            self.micro_expression = None
        if self.micro_expression is None:
            self.micro_expression = GiGAStyleOffsetModule(
                anchor_count=self._micro_expression_config["anchor_count"],
                topk=self._micro_expression_config["topk"],
                hidden_dim=self._micro_expression_config["hidden_dim"],
                embed_dim=self._micro_expression_config["embed_dim"],
                head_mode=self._micro_expression_config["head_mode"],
                appearance_mode=self._micro_expression_config["appearance_mode"],
                sh_degree=self._micro_expression_config["sh_degree"],
                position_scale=self._micro_expression_config["position_scale"],
                scaling_scale=self._micro_expression_config["scaling_scale"],
                opacity_scale=self._micro_expression_config["opacity_scale"],
                rotation_scale=self._micro_expression_config["rotation_scale"],
                appearance_scale=self._micro_expression_config["appearance_scale"],
            ).cuda()

    @staticmethod
    def _triangle_features(face_center, face_orien_mat, face_scaling):
        return torch.cat((face_center, face_orien_mat.reshape(-1, 9), face_scaling), dim=-1)

    def _compute_neutral_triangle_features(self):
        flame_param = self.flame_param
        zeros_expr = torch.zeros_like(flame_param["expr"][[0]])
        zeros_rotation = torch.zeros_like(flame_param["rotation"][[0]])
        zeros_neck = torch.zeros_like(flame_param["neck_pose"][[0]])
        zeros_jaw = torch.zeros_like(flame_param["jaw_pose"][[0]])
        zeros_eyes = torch.zeros_like(flame_param["eyes_pose"][[0]])
        zeros_translation = torch.zeros_like(flame_param["translation"][[0]])
        zeros_dynamic_offset = torch.zeros_like(flame_param["dynamic_offset"][[0]])
        verts, _ = self.flame_model(
            flame_param["shape"][None, ...],
            zeros_expr,
            zeros_rotation,
            zeros_neck,
            zeros_jaw,
            zeros_eyes,
            zeros_translation,
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param["static_offset"],
            dynamic_offset=zeros_dynamic_offset,
        )
        faces = self.flame_model.faces
        triangles = verts[:, faces]
        face_center = triangles.mean(dim=-2).squeeze(0)
        face_orien_mat, face_scaling = compute_face_orientation(
            verts.squeeze(0), faces.squeeze(0), return_scale=True
        )
        return self._triangle_features(face_center, face_orien_mat, face_scaling).detach()

    def begin_render(self):
        if self.micro_expression is None or self._xyz.shape[0] == 0:
            return
        if self.face_center is None:
            self.select_mesh_by_timestep(0)
        triangle_features = self._triangle_features(
            self.face_center, self.face_orien_mat, self.face_scaling
        )
        if self._micro_expression_config["use_deformation_features"]:
            if self._micro_expression_reference_features is None:
                self._micro_expression_reference_features = self._compute_neutral_triangle_features()
            triangle_features = triangle_features - self._micro_expression_reference_features
        self._micro_expression_offsets = self.micro_expression(
            triangle_features,
            self.binding.long(),
            self._xyz,
            self._scaling,
            self._opacity,
            self._rotation,
            self._features_dc.squeeze(1),
            self._features_rest,
        )
        self._micro_expression_regularization = sum(
            value.square().mean() for value in self._micro_expression_offsets.values()
        )

    def end_render(self):
        self._micro_expression_offsets = None

    def get_attribute_offset(self, name):
        if self._micro_expression_offsets is None:
            return None
        return self._micro_expression_offsets.get(name)

    def compute_micro_expression_offset_loss(self):
        return self._micro_expression_regularization

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.flame_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            num_verts = self.flame_model.v_template.shape[0]

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
                # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
            
            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()
            
            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}
            self._micro_expression_reference_features = None
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].cuda(),
            flame_param['rotation'].cuda(),
            flame_param['neck'].cuda(),
            flame_param['jaw'].cuda(),
            flame_param['eyes'].cuda(),
            flame_param['translation'].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param

        verts, verts_cano = self.flame_model(
            flame_param['shape'][None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        )
        self.update_mesh_properties(verts, verts_cano)
    
    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano
    
    def compute_dynamic_offset_loss(self):
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)

        if self.micro_expression is not None:
            micro_params = {
                'params': list(self.micro_expression.parameters()),
                'lr': training_args.micro_expression_lr,
                'name': 'micro_expression',
            }
            if training_args.micro_expression_only:
                for group in self.optimizer.param_groups:
                    for parameter in group['params']:
                        parameter.requires_grad_(False)
                self.optimizer = torch.optim.Adam([micro_params], lr=0.0, eps=1e-15)
                return
            self.optimizer.add_param_group(micro_params)
        elif training_args.micro_expression_only:
            raise ValueError("--micro_expression_only requires --enable_micro_expression")

        if self.not_finetune_flame_params:
            return

        # # shape
        # self.flame_param['shape'].requires_grad = True
        # param_shape = {'params': [self.flame_param['shape']], 'lr': 1e-5, "name": "shape"}
        # self.optimizer.add_param_group(param_shape)

        # pose
        self.flame_param['rotation'].requires_grad = True
        self.flame_param['neck_pose'].requires_grad = True
        self.flame_param['jaw_pose'].requires_grad = True
        self.flame_param['eyes_pose'].requires_grad = True
        params = [
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        # translation
        self.flame_param['translation'].requires_grad = True
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)
        
        # expression
        self.flame_param['expr'].requires_grad = True
        param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr)

        # # static_offset
        # self.flame_param['static_offset'].requires_grad = True
        # param_static_offset = {'params': [self.flame_param['static_offset']], 'lr': 1e-6, "name": "static_offset"}
        # self.optimizer.add_param_group(param_static_offset)

        # # dynamic_offset
        # self.flame_param['dynamic_offset'].requires_grad = True
        # param_dynamic_offset = {'params': [self.flame_param['dynamic_offset']], 'lr': 1.6e-6, "name": "dynamic_offset"}
        # self.optimizer.add_param_group(param_dynamic_offset)

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

        if self.micro_expression is not None:
            module_dir = Path(path).parent
            torch.save(self.micro_expression.state_dict(), module_dir / "micro_expression.pth")
            with open(module_dir / "micro_expression_config.json", "w") as config_file:
                json.dump(self._micro_expression_config, config_file, indent=2)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
            self._micro_expression_reference_features = None
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param = {
                # keep the static parameters
                'shape': self.flame_param['shape'],
                'static_offset': self.flame_param['static_offset'],
                # update the dynamic parameters
                'translation': flame_param['translation'],
                'rotation': flame_param['rotation'],
                'neck_pose': flame_param['neck_pose'],
                'jaw_pose': flame_param['jaw_pose'],
                'eyes_pose': flame_param['eyes_pose'],
                'expr': flame_param['expr'],
                'dynamic_offset': flame_param['dynamic_offset'],
            }
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
            self._micro_expression_reference_features = None
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]

        module_dir = Path(path).parent
        config_path = module_dir / "micro_expression_config.json"
        weights_path = module_dir / "micro_expression.pth"
        if kwargs.get("load_micro_expression", True) and config_path.exists() and weights_path.exists():
            with open(config_path) as config_file:
                self.enable_micro_expression(json.load(config_file), force=True)
            state_dict = torch.load(weights_path, map_location="cuda")
            self.micro_expression.load_state_dict(state_dict)

    def capture(self):
        captured = {"gaussian_model": super().capture()}
        if self.micro_expression is not None:
            captured["micro_expression_config"] = self._micro_expression_config
            captured["micro_expression"] = self.micro_expression.state_dict()
        return captured

    def restore(self, model_args, training_args):
        if not isinstance(model_args, dict):
            super().restore(model_args, training_args)
            return
        if "micro_expression" in model_args:
            self.enable_micro_expression(model_args["micro_expression_config"])
        super().restore(model_args["gaussian_model"], training_args)
        if "micro_expression" in model_args:
            self.micro_expression.load_state_dict(model_args["micro_expression"])
