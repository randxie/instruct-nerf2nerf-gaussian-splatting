import glob
from PIL import Image


def make_gif(frame_folder, gif_name):
    images = glob.glob(f"{frame_folder}/*.png")
    images.sort()
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save(gif_name, format="GIF", append_images=frames, save_all=True, duration=100, loop=0)


if __name__ == "__main__":
    make_gif('gaussian_splatting/output/115c9874-c/train/ours_7000/gt', 'gifs/gt.gif')
    make_gif('gaussian_splatting/output/115c9874-c/train/ours_7000/renders', 'gifs/rendered.gif')
