import odl
import sys
from odl.contrib.torch import OperatorModule as OperatorModule # Name changes depending on version
import numpy as np
import torch.nn as nn
import scipy.io as io
import h5py    # Python 3.6 for this
import torch
torch.set_default_dtype(torch.double)


class SparseMatrixModule(torch.nn.Module):
    def __init__(self, sparse_mat, in_channels, out_channels):
        """
        Returns a sparse linear operator as a pytorch module.
        Given an input tensor x of dimension (N, in_channels, W, H), we
        first reshape x to a (in_channels*H, N*W) tensor,
        multiplying it by the matrix which gives an (out_channels*H, N*W) tensor,
        and finally reshapeding the result to a (N, out_channels, W, H) tensor.

        :param sparse_mat:  Sparse matrix of shape (out_channels, in_channels) * K  for some K.
        :param in_channels:  Positive integer.
        :param out_channels:    Positive integer.
        """
        super(SparseMatrixModule, self).__init__()
        self.sparse_mat = sparse_mat
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        (dim_0, dim_1, dim_2, dim_3) = x.shape
        assert self.in_channels == dim_1
        mat_reshape = x.permute((1, 3, 0, 2)).reshape((self.in_channels * dim_3, dim_0 * dim_2))
        return self.sparse_mat.mm(mat_reshape).reshape(self.out_channels, dim_3, dim_0, dim_2).permute((2, 0, 3, 1))

    def adjoint(self):
        return SparseMatrixModule(self.sparse_mat.t(), self.out_channels, self.in_channels)

# SOLUTION TO LOW MEMORY: DON'T SAVE THE ENTIRE PSF
class PsfOp(torch.nn.Module):
    def __init__(self, psf, air_sino, kernel_size, n_views, device):
        """
        Returns Point-spread function as operator.
        shape of psf:  (n_bins, n_energies, )
        :param sparse_mat:  Sparse matrix of shape (out_channels, in_channels) * K  for some K.
        :param in_channels:  Positive integer.
        :param out_channels:    Positive integer.
        """
        super(PsfOp, self).__init__()
        ####
        psf = psf.transpose(2,1,0)[:,:,:,None]
        (b, e, c, _) = psf.shape # Bins, energies, detectors
        kernel = np.zeros((b, e, 2*c-1, 1))
        # Glue together the psf
        # bins, energies, detectors, views
        kernel[:,:,:c-1,:] = psf[:,:,:0:-1,:]
        kernel[:,:,c-1:,:] = psf[:,:,:,:]

        trim = (kernel.shape[2] - kernel_size) //2
        if trim > 0:
            kernel_new = kernel[:, :, trim:-trim,:]
            kernel_new[:,:,0,:] += kernel[:,:,:trim,:].sum(axis=2)
            kernel_new[:,:,-1,:] += kernel[:,:,-trim:,:].sum(axis=2)
            kernel = kernel_new

        kernel = torch.from_numpy(kernel).to("cuda").double()
        pad = kernel.shape[2]//2
        with torch.no_grad():
            conv = nn.Conv2d(in_channels=psf.shape[1], 
                             out_channels=psf.shape[0], 
                             kernel_size=(kernel_size,1), 
                             padding=(pad,0), 
                             padding_mode = "zeros",
                             #padding_mode='replicate',
                             bias=False).to("cuda")
            conv.weight = nn.Parameter(kernel)

        # Check width of sinogram.
        self.air_sino = torch.from_numpy(air_sino).permute(1, 0)[None,:,:,None].to(device)
        self.psf = conv.to(device)
        self.device = device
    

    def forward(self, x):
        return self.psf(x * self.air_sino)

    def adjoint(self):
        return PsfAdjOp(self)
    

class PsfAdjOp(nn.Module):
    def __init__(self, psfOp):
        kernel = psfOp.psf.weight
        (b, e, c, _) = tuple(kernel.shape)
        
        with torch.no_grad():
            conv = nn.ConvTranspose2d(in_channels = b,
                                      out_channels = e, 
                                      kernel_size = (c, 1), 
                                      padding = (c//2, 0),
                                      padding_mode = 'replicate',
                                      bias=False)
            conv.weight = nn.Parameter(kernel)
            conv.bias = torch.nn.Parameter(torch.zeros_like(conv.bias))
        self.conv = conv.to(device)
        self.device = psfOp.device
        self.air_sino = psfOp.air_sino
        
        
    def forward(y): # This is really backward :)
        return 0
        

        
# THE ABOVE SHOULD BE USED FOR HIGH SPREAD PSFs.    
class RayTfm(torch.nn.Module):
    def __init__(self, ray_tfm, device):
        """Pytorch wrapper of an odl ray transform operator on a given device.
        This is just needed since the default output of the Ray transform is on cpu"""
        super(RayTfm, self).__init__()
        self.ray_tfm = OperatorModule(ray_tfm).to(device)
        self.device = device

    def forward(self, x):
        return self.ray_tfm(x).to(self.device)


class Huber(odl.Operator):
    def __init__(self, domain, delta):
        super(Huber, self).__init__(domain=domain, range=domain, linear=False)
        self.delta = delta

    def _call(self, x, **kwargs):
        out = x.space.zero()
        x_arr = x.asarray()
        out[:] = self.huber(x_arr, self.delta)
        return out

    def derivative(self, x):
        x_array = x.asarray()
        return odl.MultiplyOperator(self.domain.element(self.huber_deriv(x_array, self.delta)))

    def curvature(self, x):
        x_array = x.asarray()
        return self.huber_optimal_curvature(x_array, self.delta)

    @ staticmethod
    def huber(x_arr, delta):
        x_sqr = x_arr ** 2
        x_abs = np.abs(x_arr)
        return np.where(x_abs <= delta, x_sqr, 2 * delta * x_abs - delta ** 2)

    @ staticmethod
    def huber_deriv(x_arr, delta):
        abs_x = np.abs(x_arr)
        return 2 * np.where(abs_x < delta, x_arr, delta * np.sign(x_arr)) * delta

    @ staticmethod
    def huber_optimal_curvature(x_arr, delta):
        abs_x = np.abs(x_arr)
        return 2 * np.where(abs_x < delta, np.ones_like(abs_x), delta / abs_x) * delta

    @ staticmethod
    def huber_second_derivative(x_arr, delta):
        abs_x = np.abs(x_arr)
        return 2 * np.where(abs_x < delta, np.ones_like(abs_x), np.zeros_like(abs_x)) * delta


def generate_settings_dict():
    matmat = io.loadmat("data/matlab/attenuation_and_spectra.mat")

    # Energy grid
    energy = matmat["eActVectorkeV"].flatten()

    # Attenuations
    material_attenuation_dict = {"bone": matmat["muBonemm_1"].flatten(),
                                 "iodine": matmat["muIodinemm_1"].flatten(),
                                 "water": matmat["muWatermm_1"].flatten(),
                                 "aluminium": matmat["muAluminummm_1"].flatten(),
                                 "air": np.zeros_like(energy)
                                 }

    # Densities (g/cm^3)
    material_density_dict = {"bone": 1.9, "iodine": 0.02, "water": 1.0, "aluminium": 2.7, "air": 0.0}

    # Emission spectrum
    # Generated in Spektr, 120 kVp
    exposure = 0.045/2 # Constant to force the maximal counts to be approximately 50 000.
    emitted_photons = exposure * matmat["backgroundSpectrumBeforePatientmm_2"].flatten()

    # Pointspread function
    psfFull = h5py.File("data/matlab/psfFull.mat")

    psf = np.asarray(psfFull['psfmm_2']).squeeze()

    # Deposited energies
    dep_energy = np.array(list(range(150)))
    bin_edges = np.array([5, 25, 34, 46, 60, 77, 90, 100, 150])
    left_edges = bin_edges[:-1]
    right_edges = bin_edges[1:]
    bin_indicators = (left_edges[None, :].repeat(150, axis=0) <= dep_energy[:, None].repeat(8, axis=1)) * \
                     (right_edges[None, :].repeat(150, axis=0) > dep_energy[:, None].repeat(8, axis=1))
    bin_sens = bin_indicators.astype(float)

    # Create joint kernel
    binned_psf = np.dot(np.transpose(psf, axes=(3, 2, 1, 0)), bin_sens)
    binned_psf_coarse = np.empty((21, 21, 131, 8), dtype=float)

    # Reduce resolution by averaging the bins. only need to save one quadrant.
    for i in range(21):
        for j in range(21):
            binned_psf_coarse[i, j, :, :] = np.mean(binned_psf[60 + 3 * i: 60 + 3 * i + 3,
                                                               60 + 3 * j: 60 + 3 * j + 3, :, :], axis=(0, 1))

    settings_dict = {"energies": energy,
                     "material_attenuations": material_attenuation_dict,
                     "emitted_photons": emitted_photons,
                     "binned_psf": binned_psf_coarse,
                     "material_densities": material_density_dict,
                     "filter_material": "aluminium",
                     "filter_thickness": 10,#13
                     "distance_source_to_detector": 1000,
                     "distance_source_to_isocenter": 500,
                     "detector_spacing": 0.5}


    torch.save(settings_dict, "data/detector_settings.dict")


def create_operator_components(img_width_pix,
                               img_width,
                               n_detectors,
                               n_views,
                               device="cuda",
                               materials=("bone", "water", "iodine"),
                               img_width_unit='fov_frac',
                               settings_data_path=None):
    """
    Return forward operator of a photon-counting CT setup, and its adjoint - the backward operator.

    :param img_width_pix:       Number of pixels in either direction of input images
    :param img_width:           Width of image, either in mm or as fraction of the fov
    :param n_detectors:         Number of detectors in the setup
    :param n_views:             Number of angles around the detector where data is collected.
    :param materials:           List of the basis materials in the input images, ordered.
    :param img_width_unit:      Spatial unit for img_width- either "fov_frac" or "mm".
    :param device:              Device on which the operators are evaluated. "cuda" or "cpu" allowed.
    :param settings_data_path   Path to settings data. The format is explained in the gen_settings function.
    :param kernel_width:        Width of the response kernel, which models cross-talk.
    :return:                    Two pytorch modules, the forward operator and its adjoint, the backward operator.
    """

    # LOAD DATA
    data_dict = torch.load(settings_data_path)

    # Physical quantities
    energies = data_dict["energies"]
    material_attenuation_dict = data_dict["material_attenuations"]
    material_density_dict = data_dict["material_densities"]

    # Detector specific data
    binned_psf = data_dict["binned_psf"]
    emitted_photons = data_dict["emitted_photons"]
    filter_material = data_dict["filter_material"]
    filter_thickness = data_dict["filter_thickness"]
    sdd = data_dict["distance_source_to_detector"]
    sid = data_dict["distance_source_to_isocenter"]
    detector_spacing = data_dict["detector_spacing"]

    n_energies = len(energies)
    n_materials = len(materials)
    n_bins = binned_psf.shape[3]

    # ===============================
    # INPUT TESTING
    # ===============================

    # Check that the quantities are discretised the same way
    assert n_energies == binned_psf.shape[2] and \
           all(n_energies == len(attenuation) for attenuation in material_attenuation_dict.values()) and \
           n_energies == len(emitted_photons), \
        "Expected material_spectra, response_kernel and source spectrum to be discretised with the same grid. "
    # Check that all selected materials are available
    assert all(m in material_density_dict.keys() and m in material_density_dict.keys() for m in materials), \
        f"Missing material attenuation data for materials."
    # Check that the filter material is available.
    assert filter_material in material_density_dict.keys(), "Missing material attenuation data for filter"
    assert device in ["cpu", "cuda"], "device must be either 'cpu' or 'cuda'."
    assert img_width_unit in ["mm", "fov_frac"], "image width unit must be 'mm' or 'fov_frac."



    # ====================
    # Ray Transform
    # ====================

    # EXTRACT REQUIRED QUANTITIES
    isoc_det_dist = sdd - sid  # Distance from isocenter to detector [mm]
    detector_pos = np.arange(n_detectors) * detector_spacing
    detector_pos -= detector_pos[-1] / 2

    if img_width_unit == 'mm':
        img_width_mm = img_width
    elif img_width_unit == 'fov_frac':
        # Calculate biggest image diagonal before clipping outside fov, divide by sqrt(2) and multiply by img_width.
        img_width_mm = 2 * img_width * np.sin(detector_pos[-1] / sdd) * sid / np.sqrt(2)
    if img_width_mm > 2 * np.sin(detector_pos[-1] / sdd) * sid / np.sqrt(2) * 1.0001:
        print(f"The phantom might clip outside the fov of the detector.", file=sys.stderr)


    # Image space
    dx = img_width_mm / 2
    x_lim = (-dx, dx)  # [mm]
    y_lim = (-dx, dx)  # [mm]
    single_img_space = odl.uniform_discr((x_lim[0], y_lim[0]), (x_lim[1], y_lim[1]), (img_width_pix, img_width_pix))

    # Geometry
    angle_partition = odl.uniform_partition(0, 2 * np.pi, n_views)
    detector_partition = odl.uniform_partition(detector_pos[0] - detector_spacing/2,
                                               detector_pos[-1] + detector_spacing/2, n_detectors)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=sid,
                                        det_radius=isoc_det_dist)  # FanBeamGeometry #ConeBeam

    # Ray transform
    ray_tfm = odl.tomo.RayTransform(single_img_space, geometry, impl="astra_" + device)
    material_ray_tfm = odl.DiagonalOperator(*n_materials * [ray_tfm])

    # ====================
    # Create forward operator
    # ====================

    # == Create material projection
    material_attenuations = np.hstack([material_attenuation_dict[material][:, None] *
                                      material_density_dict[material] for material in materials])

    # ==
    filter_attenuation = material_attenuation_dict[filter_material][None, :] * material_density_dict[filter_material]
    detector_filter_thickness = np.ones((n_detectors, 1)) * filter_thickness
    air_sinogram = np.exp(-detector_filter_thickness * filter_attenuation) * emitted_photons[None, :]
    #binned_psf_1d = binned_psf.sum(axis=1)
    binned_psf_1d = -binned_psf[:, 0, :, :] + 2 * binned_psf.sum(axis=1)

    # Necessary settings to build operators
    components = {"image_space": single_img_space,
                  "sinogram_space": ray_tfm.range,
                  "ray_transform": RayTfm(ray_tfm, device),
                  "ray_transform_adj": RayTfm(ray_tfm.adjoint, device),
                  "air_sinogram": air_sinogram,
                  "material_attenuations": material_attenuations,
                  "binned_psf_1d": binned_psf_1d,
                  "n_energies": n_energies,
                  "n_materials": n_materials,
                  "n_bins": n_bins,
                  "device": device
                  }
    # Return all quantities
    return components


def create_response_tensor_(binned_psf, air_sinogram, device, psf_width: int = None):

    (max_psf_width, n_energies, n_bins) = binned_psf.shape
    (n_detectors, _) = air_sinogram.shape

    if psf_width is None:
        psf_width = max_psf_width
    else:
        assert psf_width <= max_psf_width, "psf is too narrow for the requested kernel width."
        assert psf_width > 0, "psf must be a positive integer."

    # Loop through all energies, bins and detectors.
    # Loop throguh adjacent detectors and combine
    # the response kernel with the source and filter to form a sparse linear operator
    data = []
    indices = [[], []]
    for e in range(n_energies):  # Loop through energies
        for b in range(n_bins):  # Loop through bins
            for l in range(n_detectors):  # Loop through detectors
                dl_lo = max(-l, -max_psf_width + 1)  # Calculate lowest neighboring detector
                dl_hi = min(n_detectors - l - 1, max_psf_width - 1) + 1  # Calculate highest neighboring detector
                #print(f"Energy {e}, Bin {b}, looping {dl_hi-dl_lo} detectors from {l+dl_lo} to {l+dl_hi}")
                # Loop through neighboring detectors
                for dl in range(dl_lo, dl_hi):
                    # If the neighboring detector is closer than psf_width, allow interaction.
                    # Otherwise, add the interaction to the closest edge of the psf.
                    if abs(dl) <= psf_width-1:
                        # Add row and column indices
                        indices[0].append(b * n_detectors + l)
                        indices[1].append(e * n_detectors + l + dl)
                        # If detector is the furthest away that is allowed, sum rest of the neighbors
                        if abs(dl) == psf_width-1:
                            if dl > 0:
                                data.append((binned_psf[dl:dl_hi, e, b] * air_sinogram[l + dl: l + dl_hi, e]).sum())
                            elif dl < 0:
                                data.append((binned_psf[-dl:-dl_lo + 1, e, b][::-1] * air_sinogram[l + dl_lo:l + dl + 1, e]).sum())
                            else: # dl=0.
                                data.append((binned_psf[0:dl_hi, e, b] * air_sinogram[l:l+dl_hi, e]).sum()
                                            + (binned_psf[1:-dl_lo+1, e, b][::-1] * air_sinogram[l+dl_lo:l, e]).sum())
                        else:
                            data.append(binned_psf[abs(dl), e, b] * air_sinogram[l + dl, e])
                    # If the neighboring detector is too far away, skip it.
                    else:
                        pass
    #data = []
    #indices = [[], []]
    #for e in range(n_energies):  # Loop through energies
    #    for b in range(n_bins):  # Loop through bins
    #        for l in range(n_detectors):  # Loop through detectors
    #            dl_lo = max(-l, -max_psf_width + 1)  # Calculate lowest neighboring detector
    #            dl_hi = min(n_detectors - l, max_psf_width) # Calculate highest neighboring detector
    #            #print(f"Energy {e}, Bin {b}, looping {dl_hi-dl_lo} detectors from {l+dl_lo} to {l+dl_hi}")
    #            # Loop through neighboring detectors
    #            for dl in range(dl_lo, dl_hi):
    #                # If the neighboring detector is closer than psf_width, allow interaction.
    #                # Otherwise, add the interaction to the closest edge of the psf.
    #                if abs(dl) <= psf_width-1:
    #                    # Add row and column indices
    #                    indices[0].append(b * n_detectors + l)
    #                    indices[1].append(e * n_detectors + l + dl)
    #                    # If detector is the furthest away that is allowed, sum rest of the neighbors
    #                    if (dl==dl_lo) or (dl==dl_hi-1):#(abs(dl) == psf_width-1) or :
    #                        if dl > 0:
    #                            # Add extrapolation
    #                            tail = (binned_psf[dl_hi:, e, b]*air_sinogram[-1,e]).sum()
    #                            # Sum together with other
    #                            tail += (binned_psf[dl:dl_hi, e, b] * air_sinogram[l + dl: l + dl_hi, e]).sum()
     #                           data.append(tail)
    #                        elif dl < 0:
    #                            # Add extrapolation
    #                            tail = (binned_psf[-dl_lo+1:, e, b]*air_sinogram[0,e]).sum()
    #                            # Sum together with data inside sinogram
    #                            tail += (binned_psf[-dl:-dl_lo + 1, e, b][::-1] * air_sinogram[l + dl_lo:l + dl + 1, e]).sum()
    #                            data.append(tail)
    #                        else: # dl=0.
    #                            #print("dl=0")
    #                            tail =  (binned_psf[dl_hi:,e,b]*air_sinogram[-1,e]).sum()
    #                            tail += (binned_psf[-dl_lo+1:,e,b]*air_sinogram[0,e]).sum()
    #                            tail += (binned_psf[0:dl_hi, e, b] * air_sinogram[l:l+dl_hi, e]).sum()
    #                            tail += (binned_psf[1:-dl_lo+1, e, b][::-1] * air_sinogram[l+dl_lo:l, e]).sum()
    #                            data.append(tail)
    #                    else:
    #                        data.append(binned_psf[abs(dl), e, b] * air_sinogram[l + dl, e])
    #                # If the neighboring detector is too far away, skip it.
    #                else:
    #                    pass
    #                

    shape = (n_bins * n_detectors, n_energies * n_detectors)
    response_tensor = torch.sparse_coo_tensor(indices, data, shape).to(device)
    len(indices[0])
    len(indices[1])
    return response_tensor


def assemble_fwd_bwd_modules(components, psf_width):
    """
    Given operator components generated by create_operator_components(),
    create forward and backward operators. The number of neighboring detectors included in
    the cross-talk model is determined by psf_width.

    :param components: Dictionary generated by create_operator_components()
    :param psf_width:  Integer between 1 and the width of the point-spread function in the given data.
    :return:
    """

    ray_tfm_module = components["ray_transform"]
    ray_tfm_adj_module = components["ray_transform_adj"]
    material_attenuations = components["material_attenuations"]
    binned_psf = components["binned_psf_1d"]
    air_sinogram = components["air_sinogram"]
    n_energies = components["n_energies"]
    n_bins = components["n_bins"]
    n_materials = components["n_materials"]
    device = components["device"]

    material_module = torch.nn.Conv2d(in_channels=n_materials, out_channels=n_energies, kernel_size=1)
    material_adj_module = torch.nn.Conv2d(in_channels=n_energies, out_channels=n_materials, kernel_size=1)
    weights = torch.from_numpy(material_attenuations)[:, :, None, None]

    # These weights are constant! They must not be changed...
    with torch.no_grad():
        material_module.weight = torch.nn.Parameter(weights)
        material_module.bias = torch.nn.Parameter(torch.zeros_like(material_module.bias))
        material_adj_module.weight = torch.nn.Parameter(weights.transpose(0, 1))
        material_adj_module.bias = torch.nn.Parameter(torch.zeros_like(material_adj_module.bias))

    response_tensor = create_response_tensor_(binned_psf, air_sinogram, device, psf_width)
    response = SparseMatrixModule(response_tensor, in_channels=n_energies, out_channels=n_bins)
    response_adj = response.adjoint()

    class ForwardProjector(torch.nn.Module):
        def __init__(self):
            super(ForwardProjector, self).__init__()
            self.ray_tfm_module = ray_tfm_module
            self.material_module = material_module
            self.response = response

        def forward(self, x):
            return self.response(torch.exp(-self.material_module(self.ray_tfm_module(x))))

    class BackwardProjector(torch.nn.Module):
        def __init__(self):
            super(BackwardProjector, self).__init__()
            self.ray_tfm_module = ray_tfm_module
            self.ray_adj_module = ray_tfm_adj_module
            self.material_adj_module = material_adj_module
            self.material_module = material_module
            self.response_adj = response_adj

        def forward(self, xy):
            """
            :param xy: xy=[x, y] where x is a phantom and y is in the sinogram space.
            :return:
            """
            return self.ray_adj_module(self.material_adj_module(
                -torch.exp(-self.material_module(self.ray_tfm_module(xy[0]))) * self.response_adj(xy[1])))

    return ForwardProjector().to(device).requires_grad_(False), BackwardProjector().to(device).requires_grad_(False)


def assemble_huber_regulariser(components, reg_params, huber_param):
    """
    Element-wise Huber regulariser.

    :param reg_params:          np.ndarray of Regularisation parameters, one for each basis material.
    :param huber_param:         Parameter to use for the Huber function
    :param single_img_space:    odl.uniform_discr object representing the image space.
    :param device:              Device to evaluate regulariser on
    :return: regulariser and its derivative as two separate torch Modules.
    """
    n_materials = components["n_materials"]
    single_img_space = components["image_space"]
    device = components["device"]

    grad_single = odl.Gradient(domain=single_img_space)
    grad = odl.DiagonalOperator(*n_materials * [grad_single])
    scaling_matrix = np.empty((n_materials, n_materials), object)
    for m in range(n_materials):
        scaling_matrix[m, m] = odl.ScalingOperator(domain=grad_single.range, scalar=reg_params[m])

    scaling_op = odl.ProductSpaceOperator(scaling_matrix, domain=grad.range)
    if type(huber_param) == float:
        huber_op = Huber(domain=scaling_op.range, delta=huber_param)
    else:
        assert len(huber_param) == n_materials, "Huber param must be either a float or a list of the same size as reg_params."
        huber_op = odl.DiagonalOperator(*[Huber(domain=grad_single.range, delta=huber_p) for huber_p in huber_param])
    regulariser = odl.solvers.L1Norm(huber_op.range) * huber_op * scaling_op * grad

    return OperatorModule(regulariser).to(device), OperatorModule(regulariser.gradient).to(device)


def assemble_data_discrepancy(components):
    sinogram_pixel_area = components["sinogram_space"].weighting.const

    def sinogram_inner(x, y):
        return torch.tensordot(x, y, dims=x.dim()) * sinogram_pixel_area

    def data_discrepancy(y, y_true):
        y_log_yt_over_y = torch.where(y_true == 0, torch.zeros_like(y), y_true * torch.log(y_true / y))
        return sinogram_inner(y - y_true + y_log_yt_over_y, torch.ones_like(y_true))

    def data_discrepancy_grad(y, y_true):
        return (1 - y_true / y) * sinogram_pixel_area

    return data_discrepancy, data_discrepancy_grad, sinogram_inner

