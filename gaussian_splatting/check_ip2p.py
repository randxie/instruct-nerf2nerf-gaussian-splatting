from ip2p import InstructPix2Pix
from train_in2n import get_parser
from scene import GaussianModel, Scene
from arguments import (ModelParams, OptimizationParams, PipelineParams)
import torch

diff_device = "cuda:1"

ip2p_model = InstructPix2Pix(device=diff_device, num_train_timesteps=500)
text_embedding = ip2p_model.pipe._encode_prompt(
    "Make it winter with snow",
    device=diff_device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
)

parser = get_parser()
mp = ModelParams(parser)
args = parser.parse_args("")
args.source_path = "../data/tandt/truck"

dataset = mp.extract(args)

gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians)

viewpoint_stack = scene.getTrainCameras().copy()

viewpoint_cam = viewpoint_stack.pop(0)
original_image = viewpoint_cam.original_image.to(diff_device)

edited_image = ip2p_model.edit_image(
    text_embedding.to(diff_device),
    original_image.to(diff_device, dtype=torch.float16).unsqueeze(0),
    original_image.to(diff_device, dtype=torch.float16).unsqueeze(0),
)
