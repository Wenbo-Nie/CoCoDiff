# [ICLR 2026] CoCoDiff: Correspondence-Consistent Diffusion Model for Fine-grained Style Transfer



## Setup

Our codebase is built on ([CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) and [MichalGeyer/plug-and-play](https://github.com/MichalGeyer/plug-and-play))
and has similar dependencies and model architecture.

### Create a Conda Environment

```
conda env create -f environment.yaml
conda activate CoCo
```

### Download StableDiffusion Weights

Download the StableDiffusion weights from the [CompVis organization at Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
(download the `sd-v1-4.ckpt` file), and link them:
```
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 
```

## Run CoCoDiff

For running CoCoDiff, run:

```
python run.py --cnt <content_img_dir> --sty <style_img_dir>
```
For running default configuration in sample image files, run:
```
python run.py --cnt data/cnt --sty data/sty --gamma 0.3 --T 1.5   # high style fidelity
```

To fine-tune the parameters, you have control over the following aspects in the style transfer:

- **Attention-based style injection** is removed by the `--without_attn_injection` parameter.
- **Query preservation** is controlled by the `--gamma` parameter.
  (A higher value enhances content fidelity but may result a lack of style fidelity).
- **Attention temperature scaling** is controlled through the `--T` parameter.
- **Initial latent AdaIN** is removed by the `--without_init_adain` parameter.

### Save Precomputed Inversion Features
By default, it generates a "precomputed_feats" directory and saves the DDIM inversion feature of each input image.
This reduces the time for two DDIM inversions but requires a significant amount of storage (over 3 GB for each image).
If you encounter "no space left" error, please set the "precomputed" parameter as follows:

```
python run.py --precomputed "" # not save DDIM inversion features
```

## Citation
If this work is helpful for your research, please consider citing and star:
