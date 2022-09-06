import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as im
from learned_reconstruction.learning_session import *
import pandas as pd
import torch
import matplotlib
import matplotlib.image as im
import seaborn
import torch
from tools.plots import plot_image_channels
import odl
import pandas as pd


seaborn.set(style='ticks')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# ================
# General plots
# ================

def show_rgb_phantom(x):
    """3 material phantom"""
    ar = x.asarray()
    rgb_mat = np.eye(3)
    image = np.zeros(ar.shape[1:] + (3,))
    for i in range(3):
        image[:, :, i] = sum([rgb_mat[j, i] * ar[j, :, :] for j in range(3)])
    plt.imshow(np.transpose(image[:, ::-1, :], axes=(1, 0, 2)))


def show_sinogram(sinograms, fix_color_range=False, **kwargs):
    if fix_color_range:
        kwargs["vmin"] = 0
        kwargs["vmax"] = np.max(sinograms)

    (n_bins, n_views, n_cells) = sinograms.shape
    aspect = n_views / n_cells * .8
    s = 4
    plt.figure(figsize=(s * (n_bins // 2), s * 2))
    for b in range(n_bins):
        plt.subplot(2, n_bins // 2, b + 1)
        plt.imshow(sinograms[b].asarray().T, aspect=aspect, **kwargs)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


def plot_image_channels(images,
                        image_labels=None,
                        channel_labels=None,
                        spyglass_coords=None,
                        spyglass_color="white",
                        colorbar=False,
                        subset=None,
                        **kwargs):
    """

    :param images:              List of torch.Tensor or numpy.ndarray objects with three dimensions: channels, width, height.
                                Represents images with multiple channels.
    :param image_labels:        List of Labels, one for each image.
    :param channel_labels:      List of Labels, one for each image channel (for example: ['red', 'green', 'blue'])
    :param spyglass_coords:     Coordinates for a zoomed-in view of the images, one for each channel.
                                Should be a nested list: [[center_x, center_y, width], [...], [...]] with
                                one coordinate [center_x,center_y,width] for each channel.
    :param spyglass_color:      Color of the spyglass - see matplotlib color documentation.
    :param colorbar:            Whether or not to add a colorbar on the side of each channel.
    :param subset:              If there are many channels, specify and show only a subset of the channels.
                                Should be a list of indices, for example [0,2,4].
    :param kwargs:              Keywords for the image plots, see matplotlib.pyplot.imshow documentation.
    :return:
    """
    title = kwargs.pop("title", "Figure")

    def plot_spyglass(img, center, width, outline_region, color, **kwargs):
        # Pick subwindow
        # coords = [[xleft, xright], [yleft,yright]]
        indices = [[center[0] - width // 2, center[0] + width // 2], [center[1] - width // 2, center[1] + width // 2]]
        coords = [[indices[1][0] / 128, indices[1][1] / 128], [1 - indices[0][0] / 128, 1 - indices[0][1] / 128]]

        img_spy = img[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1]]

        # Plot Background and img
        extent = (0.55, 0.95, 0.54, 0.94)
        frame_extent = tuple([e - 0.01 * (-1) ** i for i, e in enumerate(extent)])
        # plt.imshow(np.zeros_like(img_spy), extent=tuple([e - 0.02*(-1)**i for i,e in enumerate(extent)]), cmap="Greys")
        plt.fill_between(frame_extent[0:2], 2 * [frame_extent[2]], 2 * [frame_extent[3]], color=color, zorder=1)
        plt.imshow(img_spy, extent=extent, **kwargs, zorder=2)

        if outline_region:
            box_x = np.array([coords[0][0], coords[0][0], coords[0][1], coords[0][1], coords[0][0]])
            box_y = np.array([coords[1][0], coords[1][1], coords[1][1], coords[1][0], coords[1][0]])
            plt.plot(box_x,
                     box_y,
                     # linestyle='--',
                     color=color,
                     linewidth=1)

    # Remove extra dimension
    if len(images[0].shape) == 4:
        images = [phtm[0] for phtm in images]
    if subset is not None:
        images = [phtm[subset] for phtm in images]

    num_phtms = len(images)
    num_materials = images[0].shape[0]
    scale = 1.8
    plt.figure(figsize=(scale * num_phtms, scale * num_materials))
    plt.title(title)

    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    if (vmin is None) and (vmax is None):
        predefined_color_range = False
    else:
        predefined_color_range = True

    for i, phtm in enumerate(images):
        for j in range(num_materials):
            # Pop vmin, vmax or retrieve from img
            if not predefined_color_range:
                vmin = min([phtm[j].min() for phtm in images])
                vmax = max([phtm[j].max() for phtm in images])

            plt.subplot(num_materials, num_phtms, num_phtms * j + i + 1)
            # Plot image
            img: im.AxesImage = plt.imshow(phtm[j, :, :], extent=(0, 1, 0, 1), vmin=vmin, vmax=vmax, **kwargs)
            plt.xticks([])
            plt.yticks([])
            # Operations on outer edges
            if i == 0 and channel_labels is not None:
                plt.ylabel(channel_labels[j])
            if j == num_materials - 1 and image_labels is not None:
                plt.xlabel(image_labels[i])
            if spyglass_coords is not None:
                plot_spyglass(img=phtm[j],
                              center=spyglass_coords[j][:-1],
                              width=spyglass_coords[j][-1],
                              outline_region=(i == 0),
                              color=spyglass_color,
                              vmin=vmin,
                              vmax=vmax,
                              **kwargs
                              )
            plt.xlim([0, 1])
            plt.ylim([0, 1])
    plt.tight_layout()

    return


def random_logans():
    """Plot example of some data - 4 shepp-logans and a control phantom"""
    import torch
    import matplotlib.image as im

    logans = torch.load("data/material_image_700.pt")
    idx = np.random.randint(0, len(logans), 4)
    logans = logans[idx].numpy().transpose(0, 2, 3, 1)

    logan_benchmark = im.imread("data/data_benchmark/low_high_res_phantom.jpg").astype(float) / 255

    s = 3
    fig = plt.figure(figsize=(4 * s, 2 * s))
    rgb = np.array([[1.0, 0.4, 1.0],
                    [1.0, 0.4, 0.3],
                    [1.0, 0.7, 0.0]]).T
    for i in range(4):
        plt.subplot2grid((2, 4), (i % 2, round((i + 1) / 4)), colspan=1, rowspan=1)
        logan = logans[i]
        logan_rgb = np.dot(logan, rgb)
        plt.imshow(logan_rgb)
        plt.xticks([])
        plt.yticks([])
    plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2)
    plt.imshow(np.dot(logan_benchmark, rgb))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig.savefig("data/figures/Method/shepp_logan_examples.pdf", bbox_inches="tight")
