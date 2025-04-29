import shutil
from json_process import *
import subprocess
from tqdm import tqdm

def latent_extraction(single_image: SingleImage):

    data_path = f"input_dir/{single_image.season.value}-{single_image.id:02}.jpg"
    inversion_prompt = single_image.prompt

    p = Path(data_path)
    assert p.exists()
    
    cmd = ["python", "preprocess.py",
           "--data_path", data_path,
           "--inversion_prompt", inversion_prompt,
           "--steps", "50", "--save-steps", "50",
           ]
    
    # subprocess.run(cmd, capture_output=True)
    subprocess.run(cmd)

def generate_config_yaml(single_image: SingleImage, target_season: Season) -> Path:
    """
        返回保存的 yaml 文件路径:
        ```
            yaml_path: Path
        ```
    """

    yaml_dir = Path("config_yaml")
    yaml_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = yaml_dir / f"{single_image.season.value}-{single_image.id:02}-{target_season.value}-conifg_pnp.yaml"

    content = f"""
# general
seed: 1
device: 'cuda'
output_path: 'output_dir/{single_image.season.value}-{single_image.id:02}'

# data
image_path: 'input_dir/{single_image.season.value}-{single_image.id:02}.jpg'
latents_path: 'latents_forward'

# diffusion
sd_version: '2.1'
guidance_scale: 7.5
n_timesteps: 50
prompt: "{single_image.prompt} at {target_season.value}"
negative_prompt: 

# pnp injection thresholds, ∈ [0, 1]
pnp_attn_t: 0.3
pnp_f_t: 0.5
"""
    with open(yaml_path, 'w') as f:
        f.write(content)
    
    return yaml_path

def delete_latents_forward_dir():
    latents_forward_dir = Path("latents_forward")
    if latents_forward_dir.exists():
        shutil.rmtree(latents_forward_dir)

def run_pnp(yaml_path: Path):
    cmd = ["python", "pnp.py",
           "--config_path", str(yaml_path),
           ]
    # subprocess.run(cmd, capture_output=True)
    subprocess.run(cmd)

def action(image_path: str, single_image: SingleImage):

    # latent extraction
    latent_extraction(single_image)

    seasons = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]
    for target_season in tqdm(seasons, total=4, desc="生成另外三个季节的图片", leave=False):
        if target_season == single_image.season:
            continue
        # generate config yaml
        yaml_path = generate_config_yaml(single_image, target_season)

        # run pnp
        run_pnp(yaml_path)
    
    delete_latents_forward_dir()

if __name__ == "__main__":
    meta_data = MetaData.load()
    meta_data.traverse_images(action)