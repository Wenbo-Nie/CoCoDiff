import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import pickle


import os
import pickle
import argparse
from FFM.ffm_sd import SDFeaturizer4Eval
from PIL import Image
import torch
import torch.nn.functional as F
import gc

feat_maps = []

def save_img_from_sample(model, samples_ddim, fname):
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(fname)

def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T,
                'timestep':_,
                }} for _ in range(50)]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == 'k' or ori_key[-1] == 'v':
                feat_maps[i][ori_key] = sty_feat[ori_key]
    return feat_maps


def feat_merge_optimized_weight(opt, cnt_feats, sty_feats, matches_dict, attn_features_dict, cos_sim_scores_dict,wei1,wei2, start_step=0, threshold=None):
    feat_maps = [{'config': {'gamma': opt.gamma, 'T': opt.T, 'timestep': _}} for _ in range(50)]
    
    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in cnt_feats[i].items()}
        sty_feat = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in sty_feats[i].items()}
        feat_maps[i] = {'config': feat_maps[i]['config']}
        
        for layer in [6,7,8,9,10,11]:
            q_key = f'output_block_{layer}_self_attn_q'
            k_key = f'output_block_{layer}_self_attn_k'
            v_key = f'output_block_{layer}_self_attn_v'
            if q_key in cnt_feat:
                feat_maps[i][q_key] = cnt_feat[q_key]
            if k_key in sty_feat:
                feat_maps[i][k_key] = sty_feat[k_key].clone()
            if v_key in sty_feat:
                feat_maps[i][v_key] = sty_feat[v_key].clone()
            
            matches = matches_dict[layer]
            cos_sim_scores = cos_sim_scores_dict[layer]
            attn_features = attn_features_dict[6] if layer in [6,7,8] else attn_features_dict[9]
            mul=wei1
            for idx, (cnt_pos_idx, sty_pos_idx) in enumerate(matches):
                #print("a")
                if(layer>8):
                    mul=wei2
                    threshold=0.55
                    #print("b")
                weight = cos_sim_scores[idx]
                if threshold is not None and cos_sim_scores[idx] < threshold:
                    weight = 0
                    #print("c")
                #print(f"attn_features keys: {list(attn_features.keys())}")
                #print(f"feat_maps[{i}] keys: {list(feat_maps[i].keys())}")
                if k_key in attn_features[i] and k_key in feat_maps[i]:
                    # print( f"injected std: {feat_maps[i][k_key][:, cnt_pos_idx, :].std().item():.4f}, "
                    #     )
                    feat_maps[i][k_key][:, cnt_pos_idx, :] = (
                        weight * attn_features[i][k_key][:, idx, :] + 
                        (1-weight)*feat_maps[i][k_key][:, cnt_pos_idx, :]
                    )
                    # print(f"injected std: {feat_maps[i][k_key][:, cnt_pos_idx, :].std().item():.4f}, "
                    #     )
                if v_key in attn_features[i] and v_key in feat_maps[i]:
                    # print(f"injected std: {feat_maps[i][v_key][:, cnt_pos_idx, :].std().item():.4f}, "
                    #     )
                    feat_maps[i][v_key][:, cnt_pos_idx, :] = (
                        weight * attn_features[i][v_key][:, idx, :] + 
                        (1-weight)*feat_maps[i][v_key][:, cnt_pos_idx, :]
                    )
                    # print(f"injected std: {feat_maps[i][v_key][:, cnt_pos_idx, :].std().item():.4f}, "
                    #     )

    return feat_maps

        


def feat_merge_optimized_add(opt, cnt_feats, sty_feats, matches_dict, attn_features_dict, cos_sim_scores_dict,wei1,wei2, start_step=0, threshold=None):
    feat_maps = [{'config': {'gamma': opt.gamma, 'T': opt.T, 'timestep': _}} for _ in range(50)]
    
    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in cnt_feats[i].items()}
        sty_feat = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in sty_feats[i].items()}
        feat_maps[i] = {'config': feat_maps[i]['config']}
        
        for layer in [6,7,8,9,10,11]:
            q_key = f'output_block_{layer}_self_attn_q'
            k_key = f'output_block_{layer}_self_attn_k'
            v_key = f'output_block_{layer}_self_attn_v'
            if q_key in cnt_feat:
                feat_maps[i][q_key] = cnt_feat[q_key]
            if k_key in sty_feat:
                feat_maps[i][k_key] = sty_feat[k_key].clone()
            if v_key in sty_feat:
                feat_maps[i][v_key] = sty_feat[v_key].clone()
            
            matches = matches_dict[layer]
            cos_sim_scores = cos_sim_scores_dict[layer]
            attn_features = attn_features_dict[6] if layer in [6,7,8] else attn_features_dict[9]
            mul=wei1
            for idx, (cnt_pos_idx, sty_pos_idx) in enumerate(matches):
                #print("a")
                if(layer>8):
                    mul=wei2
                    threshold=0.55
                    #print("b")
                weight = cos_sim_scores[idx]**2*mul
                if threshold is not None and cos_sim_scores[idx] < threshold:
                    weight = 0
                #print(f"attn_features keys: {list(attn_features.keys())}")
                #print(f"feat_maps[{i}] keys: {list(feat_maps[i].keys())}")
                if k_key in attn_features[i] and k_key in feat_maps[i]:
                    # print( f"injected std: {feat_maps[i][k_key][:, cnt_pos_idx, :].std().item():.4f}, "
                    #     )
                    feat_maps[i][k_key][:, cnt_pos_idx, :] = (
                        weight * attn_features[i][k_key][:, idx, :] + 
                        feat_maps[i][k_key][:, cnt_pos_idx, :]
                    )
                    # print(f"injected std: {feat_maps[i][k_key][:, cnt_pos_idx, :].std().item():.4f}, "
                    #     )
                if v_key in attn_features[i] and v_key in feat_maps[i]:
                    # print(f"injected std: {feat_maps[i][v_key][:, cnt_pos_idx, :].std().item():.4f}, "
                    #     )
                    feat_maps[i][v_key][:, cnt_pos_idx, :] = (
                        weight * attn_features[i][v_key][:, idx, :] + 
                        feat_maps[i][v_key][:, cnt_pos_idx, :]
                    )
                    # print(f"injected std: {feat_maps[i][v_key][:, cnt_pos_idx, :].std().item():.4f}, "
                    #     )
    
    return feat_maps

def feat_merge_change(opt, cnt_feats, sty_feats, matches_dict, attn_features_dict, cos_sim_scores_dict,wei1,wei2, start_step=0, threshold=None):
    feat_maps = [{'config': {'gamma': opt.gamma, 'T': opt.T, 'timestep': _}} for _ in range(50)]
    
    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in cnt_feats[i].items()}
        sty_feat = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in sty_feats[i].items()}
        feat_maps[i] = {'config': feat_maps[i]['config']}
        
        for layer in [6,7,8,9,10,11]:
            q_key = f'output_block_{layer}_self_attn_q'
            k_key = f'output_block_{layer}_self_attn_k'
            v_key = f'output_block_{layer}_self_attn_v'
            if q_key in cnt_feat:
                feat_maps[i][q_key] = cnt_feat[q_key]
            if k_key in sty_feat:
                feat_maps[i][k_key] = cnt_feat[k_key].clone()
            if v_key in sty_feat:
                feat_maps[i][v_key] = cnt_feat[v_key].clone()
            
            matches = matches_dict[layer]
            cos_sim_scores = cos_sim_scores_dict[layer]
            attn_features = attn_features_dict[6] if layer in [6,7,8] else attn_features_dict[9]
            mul=wei1
            for idx, (cnt_pos_idx, sty_pos_idx) in enumerate(matches):
                if(layer>8):
                    mul=wei2
                weight = cos_sim_scores[idx]**2*mul
                if threshold is not None and cos_sim_scores[idx] < threshold:
                    weight = 0
                if k_key in attn_features and k_key in feat_maps[i]:
                    feat_maps[i][k_key][:, cnt_pos_idx, :] = (
                        weight * attn_features[k_key][:, idx, :]
                    )
                if v_key in attn_features and v_key in feat_maps[i]:
                    feat_maps[i][v_key][:, cnt_pos_idx, :] = (
                        weight * attn_features[v_key][:, idx, :]
                    )
    
    return feat_maps


def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3],keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3],keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3],keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3],keepdim=True)
    output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
    return output

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def  extract_unet_attention(sty_feat, sty_pos_indices, attn_layers=[6,7,8,9,10,11]):
    attn_features = []
    for i in range(len(sty_feat)):
        feat_dict = {}
        for layer in attn_layers:
            for feat_type in ['q', 'k', 'v']:
                key = f'output_block_{layer}_self_attn_{feat_type}'
                if key in sty_feat[i]:
                    feat = sty_feat[i][key]
                    feat_dict[key] = feat[:, sty_pos_indices, :]
        attn_features.append(feat_dict)
    return attn_features

def clear_gpu_memory(variables):
    for var in variables:
        if isinstance(var, torch.Tensor):
            del var
        elif isinstance(var, list) or isinstance(var, dict):
            for item in var if isinstance(var, list) else var.values():
                if isinstance(item, torch.Tensor):
                    del item
    torch.cuda.empty_cache()  
    import gc
    gc.collect()  

def first(opt):


    feat_path_root = opt.precomputed

    seed_everything(22)
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)
    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)
    
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed

    global feat_maps
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T
                }} for _ in range(50)]

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    start_step = opt.start_step
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    sty_img_list = sorted(os.listdir(opt.sty))
    cnt_img_list = sorted(os.listdir(opt.cnt))

    begin = time.time()
    for sty_name in sty_img_list:
        sty_name_ = os.path.join(opt.sty, sty_name)
        init_sty = load_img(sty_name_).to(device)
        seed = -1
        sty_feat_name = os.path.join(feat_path_root, os.path.basename(sty_name).split('.')[0] + '_sty.pkl')
        sty_z_enc = None

        if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
            print("Precomputed style feature loading: ", sty_feat_name)
            with open(sty_feat_name, 'rb') as h:
                sty_feat = pickle.load(h)
                sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
        else:
            init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
            sty_z_enc, _ = sampler.encode_ddim(init_sty.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                                callback_ddim_timesteps=save_feature_timesteps,
                                                img_callback=ddim_sampler_callback)
            sty_feat = copy.deepcopy(feat_maps)
            sty_z_enc = feat_maps[0]['z_enc']


        for cnt_name in cnt_img_list:
            cnt_name_ = os.path.join(opt.cnt, cnt_name)
            init_cnt = load_img(cnt_name_).to(device)
            cnt_feat_name = os.path.join(feat_path_root, os.path.basename(cnt_name).split('.')[0] + '_cnt.pkl')
            cnt_feat = None

            # ddim inversion encoding
            if len(feat_path_root) > 0 and os.path.isfile(cnt_feat_name):
                print("Precomputed content feature loading: ", cnt_feat_name)
                with open(cnt_feat_name, 'rb') as h:
                    cnt_feat = pickle.load(h)
                    cnt_z_enc = torch.clone(cnt_feat[0]['z_enc'])
            else:
                init_cnt = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
                cnt_z_enc, _ = sampler.encode_ddim(init_cnt.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                    end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                                    callback_ddim_timesteps=save_feature_timesteps,
                                                    img_callback=ddim_sampler_callback)
                cnt_feat = copy.deepcopy(feat_maps)
                cnt_z_enc = feat_maps[0]['z_enc']

            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        # inversion
                        output_name = f"{os.path.basename(cnt_name).split('.')[0]}_stylized_{os.path.basename(sty_name).split('.')[0]}.png"

                        print(f"Inversion end: {time.time() - begin}")
                        if opt.without_init_adain:
                            adain_z_enc = cnt_z_enc
                        else:
                            adain_z_enc = adain(cnt_z_enc, sty_z_enc)
                        feat_maps = feat_merge(opt, cnt_feat, sty_feat, start_step=start_step)
                        if opt.without_attn_injection:
                            feat_maps = None

                        # inference
                        samples_ddim, intermediates = sampler.sample(S=ddim_steps,
                                                        batch_size=1,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=adain_z_enc,
                                                        injected_features=feat_maps,
                                                        start_step=start_step,
                                                        )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                        x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))

                        img.save(os.path.join(output_path, output_name))
                        if len(feat_path_root) > 0:
                            print("Save features")
                            if not os.path.isfile(cnt_feat_name):
                                with open(cnt_feat_name, 'wb') as h:
                                    pickle.dump(cnt_feat, h)
                            if not os.path.isfile(sty_feat_name):
                                with open(sty_feat_name, 'wb') as h:
                                    pickle.dump(sty_feat, h)
    clear_gpu_memory([init_sty, init_cnt, sty_z_enc, cnt_z_enc, adain_z_enc, samples_ddim, feat_maps, sty_feat, cnt_feat])
    print(f"Total end: {time.time() - begin}")

def main(cos,wei1,wei2,opt):
  


    feat_path_root = opt.precomputed
    output_feat = opt.output_feat
    os.makedirs(output_feat, exist_ok=True)

    seed_everything(22)
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)

    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)
    
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed

    global feat_maps
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T
                }} for _ in range(50)]

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks, i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map
    

    start_step = opt.start_step
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    sty_img_list = sorted(os.listdir(opt.sty))
    cnt_img_list = sorted(os.listdir(opt.cnt))

    begin = time.time()
    for sty_name in sty_img_list:
        if not sty_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        sty_name_ = os.path.join(opt.sty, sty_name)
        init_sty = load_img(sty_name_).to(device)
        sty_feat_name = os.path.join(output_feat, f"{os.path.basename(sty_name).split('.')[0]}_cnt.pkl")
        sty_z_enc = None

        if os.path.isfile(sty_feat_name):
            print(f"Precomputed style feature loading: {sty_feat_name}")
            with open(sty_feat_name, 'rb') as h:
                sty_feat = pickle.load(h)
                sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
        else:
            init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
            sty_z_enc, _ = sampler.encode_ddim(init_sty.clone(), num_steps=ddim_inversion_steps, 
                                             unconditional_conditioning=uc,
                                             end_step=time_idx_dict[ddim_inversion_steps-1-start_step],
                                             callback_ddim_timesteps=save_feature_timesteps,
                                             img_callback=ddim_sampler_callback)
            sty_feat = copy.deepcopy(feat_maps)
            sty_z_enc = feat_maps[0]['z_enc']
            if len(feat_path_root) > 0:
                with open(sty_feat_name, 'wb') as h:
                    pickle.dump(sty_feat, h)

        for cnt_name in cnt_img_list:
            if not cnt_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            cnt_name_ = os.path.join(opt.cnt, cnt_name)
            init_cnt = load_img(cnt_name_).to(device)
            cnt_feat_name = os.path.join(output_feat, f"{os.path.basename(cnt_name).split('.')[0]}_sty.pkl")
            cnt_feat = None

            if os.path.isfile(cnt_feat_name):
                print("yes")
                print(f"Precomputed content feature loading: {cnt_feat_name}")
                with open(cnt_feat_name, 'rb') as h:
                    cnt_feat = pickle.load(h)
                    cnt_z_enc = torch.clone(cnt_feat[0]['z_enc'])
            else:
                init_cnt = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
                cnt_z_enc, _ = sampler.encode_ddim(init_cnt.clone(), num_steps=ddim_inversion_steps, 
                                                 unconditional_conditioning=uc,
                                                 end_step=time_idx_dict[ddim_inversion_steps-1-start_step],
                                                 callback_ddim_timesteps=save_feature_timesteps,
                                                 img_callback=ddim_sampler_callback)
                cnt_feat = copy.deepcopy(feat_maps)
                cnt_z_enc = feat_maps[0]['z_enc']
                if len(feat_path_root) > 0:
                    with open(cnt_feat_name, 'wb') as h:
                        pickle.dump(cnt_feat, h)

            matches_dict = {}
            cos_sim_scores_dict = {}
            for up_ft_index in [1, 2]:
                matches_path = os.path.join(opt.precomputed, f'up_ft_index_{up_ft_index}', 
                                           f"{os.path.basename(cnt_name).split('.')[0]}_"
                                           f"{os.path.basename(sty_name).split('.')[0]}_stylized_{os.path.basename(cnt_name).split('.')[0]}_matches.pkl")
                cos_sim_scores_path = matches_path.replace('_matches.pkl', '_cos_sim_scores.pkl')
                if not (os.path.isfile(matches_path) and os.path.isfile(cos_sim_scores_path)):
                    print(f"Warning: Matches not found for up_ft_index={up_ft_index},{matches_path}")
                    continue
                with open(matches_path, 'rb') as h:
                    matches = pickle.load(h)
                with open(cos_sim_scores_path, 'rb') as h:
                    cos_sim_scores = pickle.load(h)
                matches_dict[up_ft_index] = matches
                cos_sim_scores_dict[up_ft_index] = cos_sim_scores
         
            layer_matches = {
                6: matches_dict[1],  # 32x32
                7: matches_dict[1],
                8: matches_dict[1],
                9: matches_dict[2],  # 64x64
                10: matches_dict[2],
                11: matches_dict[2]
            }
            layer_cos_sim_scores = {
                6: cos_sim_scores_dict[1],
                7: cos_sim_scores_dict[1],
                8: cos_sim_scores_dict[1],
                9: cos_sim_scores_dict[2],
                10: cos_sim_scores_dict[2],
                11: cos_sim_scores_dict[2]
            }
            
    
            sty_pos_indices_32 = [sty_pos_idx for _, sty_pos_idx in matches_dict[1]]
            sty_pos_indices_64 = [sty_pos_idx for _, sty_pos_idx in matches_dict[2]]
            attn_features_32 = extract_unet_attention(sty_feat, sty_pos_indices_32, attn_layers=[6,7,8])
            attn_features_64 = extract_unet_attention(sty_feat, sty_pos_indices_64, attn_layers=[9,10,11])
            attn_features_dict = {6: attn_features_32, 9: attn_features_64}
      
            del matches_dict, cos_sim_scores_dict
            del sty_pos_indices_32, sty_pos_indices_64, attn_features_32, attn_features_64
            gc.collect()
      
            adain_z_enc = adain(cnt_z_enc, sty_z_enc) if not opt.without_init_adain else cnt_z_enc
            feat_maps = feat_merge_optimized_add(opt, cnt_feat, sty_feat, layer_matches, 
                                                attn_features_dict, layer_cos_sim_scores, wei1,wei2,
                                                start_step=opt.start_step, 
                                                threshold=opt.cos_sim_threshold)
            del cnt_feat, attn_features_dict, layer_matches, layer_cos_sim_scores
            samples_ddim, _ = sampler.sample(S=opt.ddim_inv_steps, batch_size=1, shape=[opt.C, opt.H // opt.f, opt.W // opt.f],
                                            verbose=False, unconditional_conditioning=uc, eta=opt.ddim_eta,
                                            x_T=adain_z_enc, injected_features=feat_maps, start_step=opt.start_step)
            del adain_z_enc, feat_maps, cnt_z_enc
            output_name = f"cycle{cos}_{wei1}_{wei2}_{os.path.basename(cnt_name).split('.')[0]}_stylized_{os.path.basename(sty_name).split('.')[0]}.png"
            save_img_from_sample(model, samples_ddim, os.path.join(opt.output_path, output_name))
            del samples_ddim, output_name
            torch.cuda.empty_cache()
            gc.collect()

def match_features_all_pixels(featurizer, cnt_img_path, sty_img_path, img_size=[512, 512], t=261, up_ft_index=1):
    cnt_img = Image.open(cnt_img_path).convert("RGB")
    sty_img = Image.open(sty_img_path).convert("RGB")
    
   
    cnt_ft = featurizer.forward(cnt_img, img_size=img_size, t=t, up_ft_index=up_ft_index)  # [1, c, 64, 64]
    sty_ft = featurizer.forward(sty_img, img_size=img_size, t=t, up_ft_index=up_ft_index)
                                                                                                                            

    h, w = cnt_ft.shape[2], cnt_ft.shape[3]
    print("h",h,"w",w)
    cnt_ft_flat = cnt_ft.view(1, -1, h * w).permute(0, 2, 1)  # [1, 4096, c]
    sty_ft_flat = sty_ft.view(1, -1, h * w).permute(0, 2, 1)  # [1, 4096, c]
    
    matches = []
    cos_sim_scores = []
    for cnt_pos_idx in range(h * w): 
        cnt_ft_vec = cnt_ft_flat[:, cnt_pos_idx:cnt_pos_idx+1, :] 
        cos_sim = F.cosine_similarity(cnt_ft_vec, sty_ft_flat, dim=-1)  
        sty_pos_idx = cos_sim.argmax(dim=-1).item()
        matches.append((cnt_pos_idx, sty_pos_idx))
        cos_sim_scores.append(cos_sim[0, sty_pos_idx].item()) 
    

    del cnt_ft, sty_ft, cnt_ft_flat, sty_ft_flat, cos_sim
    torch.cuda.empty_cache()
    
    return matches, cos_sim_scores

def precompute_matches(cnt_dir, sty_dir, output_dir, model_id, t, up_ft_index, img_size):
    for up_ft_index in [1, 2]:
        output_dir_up = os.path.join(output_dir, f'up_ft_index_{up_ft_index}')
        os.makedirs(output_dir_up, exist_ok=True)
        featurizer = SDFeaturizer4Eval(sd_id=model_id, null_prompt='', cat_list=[])
        for sty_name in sorted(os.listdir(sty_dir)):
            if not sty_name.endswith(('.png', '.jpg', '.jpeg')):
                continue
            sty_name_ = os.path.join(sty_dir, sty_name)
            for cnt_name in sorted(os.listdir(cnt_dir)):
                if not cnt_name.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                cnt_name_ = os.path.join(cnt_dir, cnt_name)
                matches_path = os.path.join(output_dir_up, f"{os.path.basename(cnt_name).split('.')[0]}_"
                                                  f"{os.path.basename(sty_name).split('.')[0]}_matches.pkl")
                cos_sim_scores_path = matches_path.replace('_matches.pkl', '_cos_sim_scores.pkl')
                matches, cos_sim_scores = match_features_all_pixels(
                    featurizer, cnt_name_, sty_name_, img_size=img_size, t=t, up_ft_index=up_ft_index
                )
                with open(matches_path, 'wb') as h:
                    pickle.dump(matches, h)
                with open(cos_sim_scores_path, 'wb') as h:
                    pickle.dump(cos_sim_scores, h)
                clear_gpu_memory([matches, cos_sim_scores])
    del featurizer
    clear_gpu_memory([])

if __name__ == "__main__":
    # single dataset
    dataset_dir = "
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    cos=1
    wei1=0.6
    wei2=0.6
    #for root in subdirs:
    root=""
    cnt_d=""
    sty_d=""
    precompute_d="."
    tem_d=""
    output_d=""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default = sty_d)
    parser.add_argument('--sty', default = cnt_d)
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--T', type=float, default=1.5, help='attention temperature scaling hyperparameter')
    parser.add_argument('--gamma', type=float, default=0.75, help='query preservation hyperparameter')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='s', help='model config')
    parser.add_argument('--precomputed', type=str, default=precompute_d, help='save path for precomputed feature')
    parser.add_argument('--ckpt', type=str, default='', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument('--output_path', type=str, default=tem_d)
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    opt = parser.parse_args()
    first(opt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt_dir', default=cnt_d, type=str)
    parser.add_argument('--sty_dir', default=tem_d, type=str)
    parser.add_argument('--output_dir', default=precompute_d, type=str)
    parser.add_argument('--model_id', default='', type=str)
    parser.add_argument('--t', default=261, type=int)
    parser.add_argument('--up_ft_index', default=2, type=int)
    parser.add_argument('--img_size', default=512, type=int)
    args = parser.parse_args()
    
    precompute_matches(
        args.cnt_dir, args.sty_dir, args.output_dir, args.model_id, 
        args.t, args.up_ft_index, [args.img_size, args.img_size]
    )
    cos=0.4
    wei1=0.6
    wei2=0.6
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default=cnt_d)
    parser.add_argument('--sty', default=sty_d)
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--T', type=float, default=1.5, help='attention temperature scaling hyperparameter')
    parser.add_argument('--gamma', type=float, default=0.75, help='query preservation hyperparameter')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='', help='model config')
    parser.add_argument('--precomputed', type=str, default=precompute_d, help='save path for precomputed feature')
    parser.add_argument('--ckpt', type=str, default='', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument('--output_path', type=str, default=output_d)
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    parser.add_argument('--output_feat', type=str, default=precompute_d, help='save path for precomputed feature')
    parser.add_argument('--cos_sim_threshold', type=float, default=cos, help='cosine similarity threshold for optimized merge')
    opt = parser.parse_args()
    main(cos,wei1,wei2,opt)