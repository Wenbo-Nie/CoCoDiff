import torch
import torch.nn.functional as F
import numpy as np
import kornia.filters as K_filters
import kornia.color as K_color
from tqdm import tqdm

# Helper function (如果你的类中没有，这里也添加一个占位符)
def _noise_like(shape, device, repeat_noise):
    if repeat_noise:
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], 1, 1, 1)
    return torch.randn(shape, device=device)

# 如果你的make_ddim_timesteps不在类中，这里也添加一个占位符
def make_ddim_timesteps(schedule, num_timesteps, ddpm_num_timesteps):
    if schedule == "uniform":
        c = ddpm_num_timesteps // num_timesteps
        ddim_timesteps = np.asarray(list(range(0, ddpm_num_timesteps, c)))
    else:
        raise NotImplementedError
    return ddim_timesteps


class DiffusionSampler:
    def __init__(self, model, ddpm_num_timesteps, ddim_timesteps):
        self.model = model
        self.ddpm_num_timesteps = ddpm_num_timesteps
        self.ddim_timesteps = ddim_timesteps 
        
        # 为了让示例代码完整可运行，这里添加了一些假设的model属性
        # 在你的实际代码中，这些属性应该由你的model实例提供
        if not hasattr(self.model, 'betas'):
            self.model.betas = torch.linspace(0.0001, 0.02, ddpm_num_timesteps)
        if not hasattr(self.model, 'alphas_cumprod'):
            self.model.alphas_cumprod = torch.cumprod(1 - self.model.betas, dim=0)
        if not hasattr(self.model, 'alphas_cumprod_prev'):
            self.model.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.model.betas.device), self.model.alphas_cumprod[:-1]])
        if not hasattr(self.model, 'sqrt_one_minus_alphas_cumprod'):
            self.model.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.model.alphas_cumprod)
        if not hasattr(self.model, 'sqrt_alphas_cumprod'):
            self.model.sqrt_alphas_cumprod = torch.sqrt(self.model.alphas_cumprod)
        if not hasattr(self.model, 'ddim_sigmas_for_original_num_steps'):
            self.model.ddim_sigmas_for_original_num_steps = torch.zeros_like(self.model.alphas_cumprod)
        if not hasattr(self.model, 'parameterization'):
            self.model.parameterization = "eps" # 默认预测噪声

        # 确保DDIM的alphas和sigmas也已定义
        if not hasattr(self, 'ddim_alphas'):
            self.ddim_alphas = self.model.alphas_cumprod[ddim_timesteps]
            self.ddim_alphas_prev = torch.cat([torch.tensor([1.0], device=self.model.betas.device), self.ddim_alphas[:-1]])
            self.ddim_sqrt_one_minus_alphas = torch.sqrt(1.0 - self.ddim_alphas)
            self.ddim_sqrt_alphas = torch.sqrt(self.ddim_alphas)
            self.ddim_sigmas = torch.zeros_like(self.ddim_alphas)


    def make_negative_prompt_schedule(self, schedule_type, alpha_value, total_steps):
        if schedule_type == 'constant':
            return np.full(total_steps, alpha_value)
        else:
            raise NotImplementedError

    def _calculate_sobel_loss(self, generated_image_tensor, content_image_tensor):
        """
        使用 Kornia 计算 Sobel 边缘损失。
        generated_image_tensor: 浮点型张量，可以是 (B, 1, H, W) 或 (B, 3, H, W)。
        content_image_tensor: 浮点型张量，可以是 (B, 1, H, W) 或 (B, 3, H, W)。
        """
        # 注意：Kornia 期望图像像素值在 [0, 1] 范围。
        # 如果你的模型输出和 style_img 是 [-1, 1] 范围，你可能需要在此处进行归一化
        generated_image_tensor = generated_image_tensor.float()
        content_image_tensor = content_image_tensor.float()

        # 示例：将 [-1, 1] 范围的图像归一化到 [0, 1]
        # generated_image_tensor = (generated_image_tensor + 1.0) / 2.0
        # content_image_tensor = (content_image_tensor + 1.0) / 2.0
        
        if generated_image_tensor.shape[1] == 3:
            generated_image_gray = K_color.rgb_to_grayscale(generated_image_tensor)
            content_image_gray = K_color.rgb_to_grayscale(content_image_tensor)
        else:
            generated_image_gray = generated_image_tensor
            content_image_gray = content_image_tensor

        edges_gen = K_filters.sobel(generated_image_gray)
        edges_content = K_filters.sobel(content_image_gray)

        edges_gen_mag = torch.linalg.norm(edges_gen, dim=1, keepdim=True)
        edges_content_mag = torch.linalg.norm(edges_content, dim=1, keepdim=True)

        loss = F.l1_loss(edges_gen_mag, edges_content_mag)
        return loss

    def _apply_sobel_guidance(self, pred_x0_unbiased, current_noisy_img, time_step_idx, 
                              style_ref_img, style_guidance_scale, use_original_steps):
        """
        对 pred_x0_unbiased 应用 Sobel 边缘引导，并返回修正后的 pred_x0。
        Args:
            pred_x0_unbiased (torch.Tensor): 未经引导的 pred_x0 估计（从 p_sample_ddim 获得，带梯度）。
            current_noisy_img (torch.Tensor): 当前扩散过程中的噪声图像 x_t。
            time_step_idx (int): 当前时间步的索引。
            style_ref_img (torch.Tensor): 用于 Sobel 引导的参考图像（像素空间）。
            style_guidance_scale (float): Sobel 引导的强度。
            use_original_steps (bool): 是否使用原始DDPM步数。
        Returns:
            torch.Tensor: 修正后的 pred_x0。
        """
        device = pred_x0_unbiased.device
        
        # 确保 pred_x0_unbiased 是可追踪梯度的
        # .detach() 确保我们不依赖于 p_sample_ddim 内部的图，然后 .requires_grad_(True) 开启新图
        # 如果 pred_x0_unbiased 已经有梯度，可以直接用它
        pred_x0_for_grad = pred_x0_unbiased.clone().requires_grad_(True)
        
        # --- 关键：将潜在表示解码到像素空间以计算Sobel ---
        # 如果你的模型在潜在空间操作（例如Latent Diffusion Model），
        # 你需要在这里将 pred_x0_for_grad 从潜在空间解码到像素空间。
        # 例如：
        # if hasattr(self.model, 'decode_first_stage'): 
        #     current_image_for_sobel = self.model.decode_first_stage(pred_x0_for_grad)
        # else:
        #     current_image_for_sobel = pred_x0_for_grad
        current_image_for_sobel = pred_x0_for_grad # 示例：假设 pred_x0_for_grad 直接是像素值或可直接用作Sobel输入

        # 将 style_ref_img 转移到当前设备并确保浮点型
        ref_img_for_sobel = style_ref_img.to(device).float()
        
        # 计算 Sobel 损失
        sobel_loss_value = self._calculate_sobel_loss(current_image_for_sobel, ref_img_for_sobel)

        # 获取 Sobel 损失对 pred_x0_for_grad 的梯度
        sobel_grad_on_pred_x0 = torch.autograd.grad(outputs=sobel_loss_value, 
                                                     inputs=pred_x0_for_grad, 
                                                     retain_graph=False)[0]
        
        # 应用梯度修正：减去梯度以减小 Sobel 损失
        pred_x0_guided = pred_x0_unbiased - sobel_grad_on_pred_x0 * style_guidance_scale
        
        return pred_x0_guided


    def ddim_sampling(self, cond, shape, negative_conditioning=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      injected_features=None, callback_ddim_timesteps=None,
                      negative_prompt_alpha=1.0, negative_prompt_schedule='constant',
                      style_img=None,style_guidance=1.,content_guidance=1.,start_step=9999):
        
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            if isinstance(self.ddim_timesteps, torch.Tensor):
                all_ddim_timesteps_np = self.ddim_timesteps.cpu().numpy()
            else:
                all_ddim_timesteps_np = self.ddim_timesteps

            subset_end = int(min(timesteps / all_ddim_timesteps_np.shape[0], 1) * all_ddim_timesteps_np.shape[0])
            timesteps = all_ddim_timesteps_np[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        callback_ddim_timesteps_list = np.flip(make_ddim_timesteps("uniform", callback_ddim_timesteps, self.ddpm_num_timesteps))\
            if callback_ddim_timesteps is not None else np.flip(self.ddim_timesteps)

        negative_prompt_alpha_schedule = self.make_negative_prompt_schedule(negative_prompt_schedule, negative_prompt_alpha, total_steps)
        style_loss = None # 保留此变量，但不再用于 Sobel 引导

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            if index >= start_step:
                continue
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            injected_features_i = injected_features[i]\
                if (injected_features is not None and len(injected_features) > 0) else None
            negative_prompt_alpha_i = negative_prompt_alpha_schedule[i]
            
            # 调用 p_sample_ddim，它现在会返回带有梯度的 x_prev 和 pred_x0
            # 注意：p_sample_ddim 内部包含了DDIM的去噪公式
            x_prev_unbiased, pred_x0_unbiased = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                                                    negative_conditioning=negative_conditioning,
                                                                    quantize_denoised=quantize_denoised, temperature=temperature,
                                                                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                                                                    corrector_kwargs=corrector_kwargs,
                                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                                    unconditional_conditioning=unconditional_conditioning,
                                                                    injected_features=injected_features_i,
                                                                    negative_prompt_alpha=negative_prompt_alpha_i,
                                                                    style_loss=style_loss,style_guidance_scale=style_guidance,style_img=style_img,
                                                                    content_guidance_scale=content_guidance,
                                                                    )

            # --- Sobel 引导逻辑：在 p_sample_ddim 返回结果后进行 ---
            if style_guidance > 0 and style_img is not None:
                # 调用 _apply_sobel_guidance 来修正 pred_x0
                pred_x0_guided = self._apply_sobel_guidance(
                    pred_x0_unbiased=pred_x0_unbiased,
                    current_noisy_img=img, # 传入当前的噪声图像
                    time_step_idx=index,
                    style_ref_img=style_img,
                    style_guidance_scale=style_guidance,
                    use_original_steps=ddim_use_original_steps
                )
                
                # 现在，使用修正后的 pred_x0_guided 来重新计算 img (x_prev)
                # 这需要重新获取当前时间步的DDIM参数，并使用 pred_x0_guided
                
                alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas
                alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev
                sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if ddim_use_original_steps else self.ddim_sqrt_one_minus_alphas
                sigmas = self.model.ddim_sigmas_for_original_num_steps if ddim_use_original_steps else self.ddim_sigmas
                
                a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
                a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
                sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
                sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

                # 从修正后的 pred_x0_guided 和 img (x_t) 反推出一个修正的 e_t
                e_t_recalculated = (img - a_t.sqrt() * pred_x0_guided) / sqrt_one_minus_at

                dir_xt_guided = (1. - a_prev - sigma_t**2).sqrt() * e_t_recalculated
                noise = sigma_t * _noise_like(img.shape, device, repeat_noise) * temperature
                if noise_dropout > 0.:
                    noise = torch.nn.functional.dropout(noise, p=noise_dropout)
                
                # 最终的 img (x_prev) 使用修正后的 pred_x0 和 e_t
                img = a_prev.sqrt() * pred_x0_guided + dir_xt_guided + noise
                pred_x0 = pred_x0_guided # 更新 pred_x0 为引导后的版本

            else:
                # 如果没有Sobel引导，直接使用 p_sample_ddim 返回的结果
                img = x_prev_unbiased
                pred_x0 = pred_x0_unbiased

            if step in callback_ddim_timesteps_list:
                if callback: callback(i)
                if img_callback: img_callback(pred_x0, img, step)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    # !!! 关键修改：移除了 @torch.no_grad() !!!
    def p_sample_ddim(self, x, c, t, index, negative_conditioning=None,
                      repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      injected_features=None, negative_prompt_alpha=1.0,
                      style_guidance_scale=1.,style_loss=None,style_img=None, # 这些参数虽然传入但未在 p_sample_ddim 内部使用
                      content_guidance_scale=1.,
                      ):
        """
        这个函数执行一步DDIM去噪。
        现在它在带梯度的环境下运行，返回计算出的下一个图像 (x_prev) 和去噪后的 x0 估计 (pred_x0)。
        """
        b, *_, device = *x.shape, x.device

        if negative_conditioning is not None:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            uc = unconditional_conditioning
            nc = negative_conditioning

            c_in = torch.cat([nc, uc])
            e_t_negative, e_t_uncond_from_nc = self.model.apply_model(x_in, # 重命名避免歧义
                                                                t_in,
                                                                c_in,
                                                                injected_features=injected_features
                                                                ).chunk(2)

            c_in = torch.cat([uc, c])
            e_t_uncond_from_pos, e_t_pos = self.model.apply_model(x_in, 
                                                                t_in,
                                                                c_in,
                                                                injected_features=injected_features
                                                                ).chunk(2)

            e_t_tilde = negative_prompt_alpha * e_t_uncond_from_nc + (1 - negative_prompt_alpha) * e_t_negative
            e_t = e_t_tilde + unconditional_guidance_scale * (e_t_pos - e_t_tilde)
        else:
            if c is not None:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t_pos = self.model.apply_model(x_in, 
                                                            t_in,
                                                            c_in,
                                                            injected_features=injected_features
                                                            ).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t_pos - e_t_uncond) if unconditional_guidance_scale!=1 else e_t_pos
            else:
                x_in = x
                t_in = t
                c_in = unconditional_conditioning
                e_t_uncond = self.model.apply_model(x_in,
                                                            t_in,
                                                            c_in,
                                                            injected_features=injected_features
                                                            )
                e_t = e_t_uncond

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sqrt_alphas_cumprod = self.model.sqrt_alphas_cumprod if use_original_steps else self.ddim_sqrt_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # pred_x0 现在也会保留梯度信息
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        if quantize_denoised:
            if hasattr(self.model, 'first_stage_model') and hasattr(self.model.first_stage_model, 'quantize'):
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            else:
                print("Warning: quantize_denoised is True but self.model.first_stage_model.quantize is not found.")
        
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * _noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

# --- 示例用法 (请根据你的实际模型和数据进行调整) ---
if __name__ == "__main__":
    class MockDiffusionModel:
        def __init__(self):
            self.betas = torch.linspace(0.0001, 0.02, 1000)
            self.alphas_cumprod = torch.cumprod(1 - self.betas, dim=0)
            self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.betas.device), self.alphas_cumprod[:-1]])
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.ddim_sigmas_for_original_num_steps = torch.zeros_like(self.alphas_cumprod)
            self.parameterization = "eps"

        def apply_model(self, x, t, c, injected_features=None):
            # 模拟模型输出噪声，确保输出张量要求梯度
            return torch.randn_like(x) 

        def q_sample(self, x0, t):
            alpha_prod_t = self.alphas_cumprod[t].view(x0.shape[0], 1, 1, 1)
            sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
            sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
            noise = torch.randn_like(x0)
            return sqrt_alpha_prod_t * x0 + sqrt_one_minus_alpha_prod_t * noise

        # 如果你是Latent Diffusion Model，你将需要这样的解码器
        # def decode_first_stage(self, latents):
        #     return latents * 0.5 + 0.5 


    mock_model = MockDiffusionModel()
    ddim_timesteps_array = np.linspace(0, 999, 50, dtype=int) 

    sampler = DiffusionSampler(model=mock_model, 
                               ddpm_num_timesteps=1000, 
                               ddim_timesteps=ddim_timesteps_array)

    batch_size = 1
    channels = 4 
    height = 64
    width = 64
    shape = (batch_size, channels, height, width) 

    cond_dim = 768 
    cond = torch.randn(batch_size, 77, cond_dim) 
    uncond_cond = torch.randn(batch_size, 77, cond_dim)

    ref_img_h = height * 8 
    ref_img_w = width * 8  
    sobel_reference_image = torch.randn(batch_size, 3, ref_img_h, ref_img_w) 

    print("Starting DDIM sampling with Kornia Sobel guidance (faithful to original p_sample_ddim structure)...")
    final_generated_image, intermediates = sampler.ddim_sampling(
        cond=cond,
        shape=shape,
        unconditional_guidance_scale=7.5,
        unconditional_conditioning=uncond_cond,
        style_img=sobel_reference_image, 
        style_guidance=5.0, # 如果设为0，则无Sobel引导
        start_step=0 
    )

    print("\nSampling finished.")
    print(f"Final image shape (latent): {final_generated_image.shape}")