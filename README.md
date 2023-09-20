# instruct-nerf2nerf-gaussian-splatting

An attempt to see if the [instruct-nerf2nerf](https://github.com/ayaanzhaque/instruct-nerf2nerf) idea works with [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

## Set up

Use conda environment for local development, and docker to build the Python wheels / run colmap. Benefits:

- Can still use neovim for development and do not require remote-container support
- Only need to install nvidia-driver, do not need to install CUDA in the system path

Credits go to authors of instruct-nerf2nerf and gaussian-splatting. I mostly ported and edited from their codebase.

## Prepare Data

Download bear.zip from [this google drive](https://drive.google.com/drive/folders/1v4MLNoSwxvSlWb26xvjxeoHpgjhi_s-s). And put it into [data/](./data) folder. Use colmap to get camera pose.

You can refer to [run_colmap.sh](./run_colmap.sh) for colmap commands.

## How to Run

1. Go to [gaussian_splatting](./gaussian_splatting) folder
2. Run `python train.py -s ../data/bear` (where you put the data). It makes sure you can run gaussian splatting correctly
3. Run `python train_in2n.py -s ../data/bear`. It will run gaussian splatting, and edit the image using instruct-pix2pix every 20 iterations.

## Result @ 7000 iterations

Ground Truth

<img src="gifs/gt.gif" height="342"/> 


Rendered

<img src="gifs/rendered.gif" height="342"/>


Some findings:

- A good instruct-pix2pix model is very important. During the optimization stage, I can see some edited images are very off..
- The colors of tree leaves get brighter. Need more targeted edits in the diffusion model.
- The idea of instruct-nerf2nerf is very cool!! I am surprised by the final results. Despite rendering at some angles can still generate weird images, it overall regularizes the editing pretty well.

