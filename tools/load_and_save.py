import numpy as np
import scipy.io as io
import h5py    # Python 3.6 for this
import torch.utils.data as tdata
import pandas as pd
import torch


def add_air_as_material(phantom: torch.Tensor):
    """
    Given a phantom consisting of some basis materials,
    add and automatically fill in a material channel for the air.

    :param phantom: (N, C, W, H) tensor with C basis materials.
    :return: (N, C+1, W, H) tensor with the air material in channel C+1.
    """
    (s0, s1, s2, s3) = phantom.shape
    phantom_with_air = torch.zeros((s0, s1 + 1, s2, s3))
    phantom_with_air[:, :-1, :, :] = phantom
    phantom_with_air[:, -1, :, :] = 1 - phantom.sum(axis=1)

    return phantom_with_air


def spektr_data_to_dict():
    matmat = io.loadmat("data/matlab/attenuation_and_spectra.mat")

    atten = {}
    atten["bone"] = matmat["muBonemm_1"].flatten()
    atten["iodine"] = matmat["muIodinemm_1"].flatten()
    atten["water"] = matmat["muWatermm_1"].flatten()
    atten["aluminium"] = matmat["muAluminummm_1"].flatten()
    atten["air"] = np.zeros_like(atten["aluminium"])

    data = {}
    data["energy"] = matmat["eActVectorkeV"].flatten()
    data["emission_spectrum"] = matmat["backgroundSpectrumBeforePatientmm_2"].flatten()
    data["attenuations"] = atten


    torch.save(data, "data/spektr_data.dict")


def gen_response_kernel(directory):
    # Load psf function
    psfFull = h5py.File("data/matlab/psfFull.mat")
    x = np.asarray(psfFull['psfSamplePointsxmm'])
    y = np.asarray(psfFull['psfSamplePointsymm'])
    psf = np.asarray(psfFull['psfmm_2']).squeeze()
    #psf = psf[19:, :, :, :]

    # Load bin sensitivity
    #bin_sens = pd.read_csv("data/spectra/bin_spectra.csv").drop("energy", axis=1).values

    # HOTFIX FOR ENERGY BINS, HARD CODING
    #energy = pd.read_csv("data/spectra/bin_spectra.csv")["energy"].values
    energy = np.array(list(range(150)))
    #bin_edges = np.array([20, 25, 34, 46, 60, 77, 90, 100, 150])
    bin_edges = np.array([0, 25, 34, 46, 60, 77, 90, 100, 150])
    left_edges = bin_edges[:-1]
    right_edges = bin_edges[1:]
    #bin_indicators = (left_edges[None,:].repeat(131, axis=0) <= energy[:,None].repeat(8, axis=1)) * \
    #                 (right_edges[None, :].repeat(131, axis=0) > energy[:, None].repeat(8, axis=1))
    bin_indicators = (left_edges[None, :].repeat(150, axis=0) <= energy[:, None].repeat(8, axis=1)) * \
                     (right_edges[None, :].repeat(150, axis=0) > energy[:, None].repeat(8, axis=1))
    bin_sens = bin_indicators.astype(float)
    # END OF HOTFIX

    #bin_indicators /= bin_indicators.sum(axis=0)[None, :].repeat(131, axis=0)
    # Load spectra
    spectra = pd.read_csv("data/spectra/source_spectra.csv").drop("energy", axis=1).values.reshape(131, 1, 1, 1)
    # Create joint kernel
    psf_bin = np.dot(np.transpose(psf, axes=(1, 2, 3, 0)), bin_sens)
    # Uncomment spectra to incorporate it in the psf
    psf_bin_source = psf_bin #* spectra
    psf_bin_source = np.transpose(psf_bin_source, axes=(1, 2, 0, 3))
    #psf_bin_source_avg = np.empty((41, 41, 131, 8), dtype=float)
    psf_bin_source_avg = np.empty((21, 21, 131, 8), dtype=float)
    # Reduce resolution by averaging the bins. only need to save one quadrant.
    for i in range(21):
        for j in range(21):
            psf_bin_source_avg[i, j, :, :] = np.mean(psf_bin_source[61 + 3*i: 61 + 3*i + 3,
                                                                    61 + 3*j: 61 + 3*j + 3, :, :], axis=(0, 1))

    #response = {"bins": 8, "energies": list(range(20, 151)), "x": x, "y": y, "response_kernel": psf_bin_source_avg}
    #np.save("data/spectra/response_kernel.npy", psf_bin_source_avg, allow_pickle=True)
    np.save("data/spectra/response_kernel_bugfix_no_spectra_150.npy", psf_bin_source_avg, allow_pickle=True)
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(8):
        plt.plot(energy[19:], psf_bin_source_avg[0,0,:,i])

    for i in range(1, 131, 20):
        plt.figure()
        plt.imshow(psf_bin_source_avg[:, :, i, 2], vmin=0, vmax=np.max(psf_bin_source_avg[:, :, i, 2]))


def get_shepp_logan(idx, path="data/material_image_700.pt"):
    # Other path: data/data_train_test/material_image_600.pt
    phantoms = torch.load(path) # Material images
    return phantoms[idx, :, :, :]


def split_and_save_shepp_logans():
    # CURRENTLY SEEDED WITH SEED = 0.0

    import torch
    SEED = 0.0
    torch.manual_seed(SEED)
    phantoms = torch.load("data/material_image_700.pt")
    #train, test = tdata.random_split(phantoms, lengths=(600, 100))
    train_indices = list(range(0, 600))
    test_indices = list(range(600, 700))
    torch.save(phantoms[train_indices], "data/data_train_test/phantoms_600_det.pt")
    torch.save(phantoms[test_indices], "data/data_secret/phantoms_100_det.pt")
    np.save("data/data_train_test/indices_det.npy", train_indices)
    np.save("data/data_secret/indices_det.npy", test_indices)
    #torch.save(phantoms[train.indices], "data/data_train_test/phantoms_600_det.pt")
    #torch.save(phantoms[test.indices], "data/data_secret/phantoms_100_det.pt")

    test_train = torch.load("data/data_train_test/phantoms_600.pt")
    train = test_train[0:400]
    test = test_train[400:]
    torch.save(test, "data/sessions/data_test/phantoms_test_200.Tensor")
    #torch.save(test, "data/sessions_summer/data_test/phantoms_test_200.Tensor")
    torch.save(train, "data/sessions/data_target/phantoms_train_400.Tensor")


    data = torch.load("data/sessions/data_test/phantoms_test_200.Tensor")
    for i, dat in enumerate(data):
        torch.save(dat[None, :], f"data/sessions/data_test/phantom_test_{i}.Tensor")

    from operators.learning_tools import TvDataSet, DataPoint

    return None


def add_iodine_to_water(tensor_dir):
    data = torch.load(tensor_dir + ".Tensor")
    data[:, 1, :, :] = data[:, 2, :, :] + data[:, 1, :, :]
    torch.save(data, tensor_dir + "_joined_water_iodine.Tensor")


def save_individual_phantoms(dir, tensor_dir, serial_name):
    data = torch.load(dir + tensor_dir)
    for i in range(data.shape[0]):
        torch.save(data[None, i].clone(), f"{dir}{serial_name}_{i}.Tensor")


#add_iodine_to_water("data/sessions_summer/data_test/phantoms_test_200")
#save_individual_phantoms("data/sessions_summer/data_test/", "phantoms_train_200_joined_water_iodine.Tensor", "phantom_test")
#save_individual_phantoms("data/sessions_summer/data_target/", "phantoms_train_400_joined_water_iodine.Tensor", "target_phantom")
#x = torch.load("data/sessions/data_test/phantom_test_0.Tensor")
