import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
import matplotlib.image as im
import seaborn

seaborn.set(style='ticks')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def plot_image_channels(images,
                        image_labels=None,
                        channel_labels=None,
                        spyglass_coords=None,
                        spyglass_color="white",
                        colorbar=False,
                        subset=None,
                        scale=1.5,
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
    def plot_spyglass(img, center, width, outline_region, outline_thickness, color, **kwargs):
        # Pick subwindow
        # coords = [[xleft, xright], [yleft,yright]]
        indices = [[center[0]-width//2, center[0]+width//2], [center[1]-width//2, center[1]+width//2]]
        coords = [[indices[1][0]/128, indices[1][1]/128], [1-indices[0][0]/128, 1-indices[0][1]/128]]


        img_spy = img[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1]]

        # Plot Background and img
        extent = (0.55, 0.95, 0.54, 0.94)
        frame_extent = tuple([e - 0.01*(-1)**i for i,e in enumerate(extent)])
        #plt.imshow(np.zeros_like(img_spy), extent=tuple([e - 0.02*(-1)**i for i,e in enumerate(extent)]), cmap="Greys")
        plt.fill_between(frame_extent[0:2], 2*[frame_extent[2]], 2*[frame_extent[3]], color=color, zorder=1)
        plt.imshow(img_spy, extent=extent, **kwargs, zorder=2)

        if outline_region:
            box_x = np.array([coords[0][0], coords[0][0], coords[0][1], coords[0][1], coords[0][0]])
            box_y = np.array([coords[1][0], coords[1][1], coords[1][1], coords[1][0], coords[1][0]])
            plt.plot(box_x,
                     box_y,
                     # linestyle='--',
                     color=color,
                     linewidth=outline_thickness)

    # Remove extra dimension
    if len(images[0].shape)==4:
        images = [phtm[0] for phtm in images]
    if subset is not None:
        images = [phtm[subset] for phtm in images]

    num_phtms = len(images)
    num_materials = images[0].shape[0]
    fig, axes = plt.subplots(figsize=(scale * (num_phtms+1), scale * num_materials),nrows=num_materials, ncols=num_phtms)
    #plt.title(title)
    img_vec=[[],] * num_materials

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

            #plt.subplot(num_materials, num_phtms, num_phtms*j + i + 1)
            if num_materials == 1:
                plt.sca(axes[i])
            else:
                plt.sca(axes[j][i])
            # Plot image
            img: im.AxesImage = plt.imshow(phtm[j, :, :], extent=(0, 1, 0, 1), vmin=vmin, vmax=vmax, **kwargs)
            img_vec[j].append(img)
            plt.xticks([])
            plt.yticks([])
            # Operations on outer edges
            if i == 0 and channel_labels is not None:
                plt.ylabel(channel_labels[j])
            if j == num_materials-1 and image_labels is not None:
                plt.xlabel(image_labels[i])
            if spyglass_coords is not None:
                plot_spyglass(img=phtm[j],
                              center=spyglass_coords[j][:-1],
                              width=spyglass_coords[j][-1],
                              outline_region=True,#(i == 0),
                              color=spyglass_color,
                              vmin=vmin,
                              vmax=vmax,
                              outline_thickness=(1 if (i == 0) else 0.4),
                              **kwargs
                              )
            plt.xlim([0, 1])
            plt.ylim([0, 1])

    #plt.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.colorbar(img, ax=axes.ravel().tolist())
    return fig

def random_logans():
    """Plot example of some data - 4 shepp-logans and a control phantom"""
    import torch
    import matplotlib.image as im

    logans = torch.load("data/material_image_700.pt")
    idx = np.random.randint(0, len(logans), 4)
    logans = logans[idx].numpy().transpose(0, 2, 3, 1)

    logan_benchmark = im.imread("data/data_benchmark/low_high_res_phantom.jpg").astype(float)/ 255

    s = 3
    fig = plt.figure(figsize=(4*s, 2*s))
    rgb = np.array([[1.0, 0.4, 1.0],
                    [1.0, 0.4, 0.3],
                    [1.0, 0.7, 0.0]]).T
    for i in range(4):
        plt.subplot2grid((2, 4), (i % 2, round((i+1)/ 4)), colspan=1, rowspan=1)
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


# TESTING FUNCTIONS
def test_img_channel_plot():
    plot_image_channels([x for x in
                         torch.load("data/sessions_summer/data_target/phantoms_train_400_joined_water_iodine.Tensor")[
                         0:5].cpu().numpy()],
                        ["Image " + str(i) for i in range(5)],
                        ["Bone", "Water", "Iodine"],
                        spyglass_coords=[[50, 30, 10], [50, 30, 10], [70, 30, 10]],
                        spyglass_color=[1, 0.5, 0.2],
                        vmin=0,
                        vmax=1,
                        cmap="bone"
                        )

    plot_image_channels([torch.load(f"data/sessions/data_target/target_sinogram_{i}.Tensor").cpu().numpy() for i in range(5)],
                        ["Image " + str(i) for i in range(5)],
                        ["Bin " + str(i) for i in range(8)],
                        spyglass_coords=[[50, 50, 10]] * 8,
                        spyglass_color=[1, 0.5, 0.2],
                        cmap="bone",
                        )


    plot_image_channels([x for x in torch.load("data/data_train_test/phantoms_600_det.pt")[0:5].cpu().numpy()],
                        ["Image " + str(i) for i in range(5)],
                        ["Bone", "Water", "Iodine"],
                        spyglass_coords=[[50, 30, 10], [50, 30, 10], [70, 30, 10]],
                        spyglass_color=[1, 0.5, 0.2],
                        vmin=0,
                        vmax=1,
                        cmap="bone"
                        )

    plot_image_channels([x for x in torch.load("data/data_train_test/phantoms_600.pt")[0:5].cpu().numpy()],
                        ["Image " + str(i) for i in range(5)],
                        ["Bone", "Water", "Iodine"],
                        spyglass_coords=[[50, 30, 10], [50, 30, 10], [70, 30, 10]],
                        spyglass_color=[1, 0.5, 0.2],
                        vmin=0,
                        vmax=1,
                        cmap="bone"
                        )

