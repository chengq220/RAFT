import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.tile(img[...,None], (1, 1, 3))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(img, flo, save = None):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    
    if not save:
        cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
        cv2.waitKey()
    else:
        img_flo_bgr = (img_flo[:, :, [2, 1, 0]]).astype(np.uint8)  # Convert back to BGR and uint8
        cv2.imwrite(save, img_flo_bgr)


def flow_to_vecfield(t1, t2, flow_uv, step = 1, save = None):
    """
    Visualizes two images and a vector field side by side.

    Parameters:
    - t1, t2: 2D arrays or images to display.
    - field: A tuple (x, y, u, v) where x, y are coordinates and u, v are vector components.
    - save: Optional file path to save the plot.
    """

    # Make sure that the input images are single images
    assert len(t1.shape) == 4 and len(t2.shape) == 4
    # Make sure the 4 components of a single vector field are present
    assert len(flow_uv.shape) == 4 and flow_uv.shape[1] == 2 
    t1, t2, flow_uv = t1.squeeze().cpu().numpy(), t2.squeeze().cpu().numpy(), flow_uv.squeeze().cpu().numpy()
    xx, yy = np.meshgrid(np.arange(t1.shape[1]), np.arange(t1.shape[2]))
    grid = np.transpose(np.stack((xx, yy), axis=-1), (2,0,1))

    fig, axs = plt.subplots(1, 4, figsize=(15, 5)) 

    # Display image at time series 1
    axs[0].imshow(np.transpose(t1, (1,2,0)), cmap='gray')
    axs[0].set_title("Image 1")
    axs[0].axis('off')

    # Display image at time series 2
    axs[1].imshow(np.transpose(t2, (1,2,0)), cmap='gray')
    axs[1].set_title("Image 2")
    axs[1].axis('off')

    # Display vector field
    x, y = grid
    u, v = flow_uv
    
    axs[2].quiver(x[::step, ::step], y[::step, ::step],  u[::step, ::step],  v[::step, ::step], color='b', angles='xy', scale_units='xy', scale=1)
    axs[2].set_title("Vector Field")
    axs[2].axis('equal')

    # Calculate the magnitude of the vectors
    magnitude = np.sqrt(u**2 + v**2)
    quiver = axs[3].quiver(x, y, u, v, magnitude, angles='xy', scale_units='xy', scale=1, cmap='viridis')
    axs[3].set_title("Vector Field - Color")
    axs[3].axis('equal')

    cbar = plt.colorbar(quiver, ax=axs[3])
    cbar.set_label('Vector Magnitude')

    plt.tight_layout()

    if save:
        plt.savefig(save)
    else:
        plt.show()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        frame1 = glob.glob(os.path.join(args.path, "frame_1", '*.png'))
        frame2 = glob.glob(os.path.join(args.path, "frame_2", '*.png'))
        
        frame1 = sorted(frame1)
        frame2 = sorted(frame2)
        
        for imfile1, imfile2 in zip(frame1, frame2):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # viz(image1, flow_up, args.save)
            flow_to_vecfield(image1,image2, flow_up, step=10, save = args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--save', help="save path for inference")
    args = parser.parse_args()

    demo(args)
