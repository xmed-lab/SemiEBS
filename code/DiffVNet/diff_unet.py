import torch
import torch.nn as nn
# from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
# from light_training.utils.files_helper import save_new_model_and_delete_last
# import argparse
# import yaml
from .basic_unet import BasicUNetEncoder
from .basic_unet_denose import BasicUNetDe, BasicUNetDe_stu
from .guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from .guided_diffusion.respace import SpacedDiffusion, space_timesteps
from .guided_diffusion.resample import UniformSampler



class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, 1, 2)

        self.model = BasicUNetDe(3, 14, 13,
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        self.stu_model = BasicUNetDe_stu(3, 1, 14,
                                 act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)


    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).cuda()
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "student":
            embeddings = self.embed_model(image)
            return self.stu_model(image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (2, 13, 64, 128, 128), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out

        # elif pred_type == "ddim_sample_stu":
        #     embeddings = self.embed_model(image)
        #     # print(image.shape[0])
        #     sample_out = self.sample_diffusion.ddim_sample_loop(self.stu_model, (image.shape[0], 1, 64, 128, 128), model_kwargs={"image": image, "embeddings": embeddings})
        #     sample_out = sample_out["pred_xstart"]
        #     return sample_out



