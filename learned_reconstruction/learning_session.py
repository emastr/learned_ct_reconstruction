import torch.utils.data as tdata
from learned_reconstruction.operator_modules import create_operator_components, assemble_huber_regulariser, assemble_fwd_bwd_modules
from tools.plots import plot_image_channels
from learned_reconstruction.unet import ResUnet
import pandas as pd
import torch
import torch.nn as nn
import os
import odl
import re
import numpy as np
from tools.logger import EventTracker


class NetSum(nn.Module):
    """
    Sum a list of modules with same domain and range
    """

    def __init__(self, *models):
        super(NetSum, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return sum([model(x) for model in self.models])


class DataPoint():
    """
    An iteration step in the TV descent.
    """

    def __init__(self,
                 x: torch.Tensor,
                 y_acc: torch.Tensor,
                 y_apx: torch.Tensor,
                 yadj_acc: torch.Tensor,
                 id: int,
                 iter: int = 0):
        """
        :param x:  material image (phantom) as [n1, n2, n3] tensor
        :param y_acc: fwd_acc(x) as [m1, m2, m3] torch tensor
        :param y_apx:  fwd_apx(x) as [m1, m2, m3] torch tensor
        :param yadj_acc:  tv_loss.gradient(x)  as [n1, n2, n3] tensor
        :param id:  Id number of target phantom
        :param iter:  The number of steps
        """
        #
        self.iter = iter  # Iteration number. The true phantom has iter None.
        self.id = id  # Index pointing to the place in original data
        self.x = x  # Phantom
        # Reference for forward step
        self.y_acc = y_acc  # F_acc(x)
        self.y_apx = y_apx  # F_apx(x)
        # Reference for backward step
        self.yadj_acc = yadj_acc  # DF_acc*(x)  dLoss^theta(x)
        # These are not used until the next iterate is produced
        # New approach: skip a lot of steps so saving 10 previous iterations on the same file is prefferable
        self.dx = []
        self.ddLoss = []
        self.dLoss = []
        # self.dx = None              # alpha * P = alpha * bfgs_step(C_theta(y_adj_apx), ddLosses, dLosses)
        # self.ddLoss = None          #
        # self.dLoss = None           #

    def dict(self):
        return {"x": self.x, "y_acc": self.y_acc, "y_apx": self.y_apx, "id": self.id, "iter": self.iter}

    def save(self, path):
        """
        Save datapoint in specified directory
        :param path: path to save the datapoint at
        :return:
        """
        torch.save(self, path)
        return 0

    def to(self, device):
        self.x = self.x.to(device)
        self.y_acc = self.y_acc.to(device)
        self.y_apx = self.y_apx.to(device)
        self.yadj_acc = self.yadj_acc.to(device)
        return self

    @staticmethod
    def dataPoint_from_ops(x, fwd_accurate, fwd_approx, fwd_adj_acc, likelihood_grad, y_true, id, iter):
        """
        Create datapoint from operators
        :param x:
        :param fwd_accurate:
        :param fwd_approx:
        :param fwd_adj_acc:
        :param id:
        :param iter:
        :return:
        """
        y_acc = fwd_accurate(x)
        y_apx = fwd_approx(x)
        yadj_acc = fwd_adj_acc([x, likelihood_grad(y_acc, y_true)])
        return DataPoint(x=x,
                         y_acc=y_acc,
                         y_apx=y_apx,
                         yadj_acc=yadj_acc,
                         id=id,
                         iter=iter)


class TvDataSet(tdata.Dataset):
    def __init__(self, path, verbose=False, device="cuda"):
        """
        Instansiate a Dataset containing total variation trajectories
        at a given path. The path should not be altered through any method
        other than the methods of the dataset once it has been initialised.

        :param path: Path to the folder where the datapoints are located.
                        Should end in "/".
        """
        super(TvDataSet, self).__init__()

        # Format function
        self.verbose = verbose
        self.FORMAT = lambda idx: f"data_{idx}.DataPoint"
        RE_FORMAT = self.FORMAT("[0-9]*\\")  # perform regex search of directory.

        # Check number of files currently in directory
        # Index starts at 0.
        # Set path

        self.path = path
        self.length = len([file for file in os.listdir(path) if re.fullmatch(RE_FORMAT, file) is not None])
        self.id_iter_to_idx = {}

        # Loop through files, add to dictionary. Crucial for later analysis since indices will be random
        for idx in range(self.length):
            temp = torch.load(self.path + self.FORMAT(idx))
            assert isinstance(temp, DataPoint), "Faulty data format. Has to be DataPoint object."
            self.id_iter_to_idx[(temp.id, temp.iter)] = idx

        self.print(f"Found {len(self)} data points. Returning data object.")

        self.device = device

    def __getitem__(self, index):
        assert index < self.length, f"Index {index} out of range."
        return torch.load(self.path + f"data_{index}.DataPoint").to(self.device)

    def __len__(self):
        return self.length

    def append(self, data: DataPoint):
        # Adding datapoint
        data.save(self.path + self.FORMAT(self.length))
        self.id_iter_to_idx[(data.id, data.iter)] = self.length
        self.length += 1
        return 0

    def append_many(self, data_list: list):
        for data in data_list:
            self.append(data)
        return 0

    def load_from_id_idx(self, id, idx):
        assert (id, idx) in self.id_iter_to_idx.keys(), f"Datapoint with id {id} and index {idx} is not in the data."
        return self.__getitem__(self.id_iter_to_idx[(id, idx)])

    def print(self, msg, **kwargs):
        if self.verbose:
            print(msg, **kwargs)


class FwdBwdSession():
    """
    Session class for training and evaluating a bwd-forward correction.
    """

    def __init__(self,
                 session_name: str,
                 data_loading_path: str,
                 session_path: str,
                 learning_rate=(1e-3, 1e-3),
                 num_data=None,
                 device="cpu",
                 verbose=False,
                 validate_every_n=100,
                 save_every_n=1,
                 batch_norm=(0, 0),
                 activations=(nn.ReLU, nn.ReLU),
                 in_ch=(8, 3),
                 bfgs_stepsize=1.0,
                 reg_scale=1.0,
                 grad_clip=(None, None),
                 detector_settings_path="data/detector_settings.dict"
                 ):
        """
        :param session_name:
        :param data_loading_path:
        :param session_path:
        :param learning_rate:
        :param num_data:
        :param device:
        :param verbose:
        :param validate_every_n:
        :param save_every_n:
        """

        self.static_settings = {"session_name": session_name,
                                "session_path": session_path,
                                "batch_norm": batch_norm,
                                "activations": activations,
                                "in_ch": in_ch,
                                "detector_settings_path": detector_settings_path,
                                }

        self.dynamic_settings = {"data_loading_path": data_loading_path,
                                 "learning_rate": learning_rate,
                                 "num_data": num_data,
                                 "device": device,
                                 "verbose": verbose,
                                 "validate_every_n": validate_every_n,
                                 "save_every_n": save_every_n,
                                 "bfgs_stepsize": bfgs_stepsize,
                                 "reg_scale": reg_scale,
                                 "grad_clip": grad_clip
                                 }

        self.verbose = verbose
        self.device = device
        torch.device(device)

        # Data paths
        self.data_init_path = session_path + "data_init/"
        self.data_target_path = session_path + "data_target/"
        self.data_test_path = session_path + "data_test/"
        self.name = session_name
        self.session_path = session_path + session_name + "/"
        self.session_states_path = self.session_path + "session_states/"
        self.session_data_iter_path = data_loading_path + session_name + "/session_data/"
        self.log_path = self.session_path + "session_log.txt"
        self.script_path = self.session_path + "scripts/"
        is_old_session = self.makedirs()  # Initialise the above direcories

        # # Operators
        # Define forward operators
        # All of this hassle is to insure compatibility with the ODL implementation
        # I.e. , obtaining the same results.

        self.print("Creating operators...")
        img_width_pix = 256
        n_detectors = 512
        n_views = 512
        components = create_operator_components(img_width_pix=img_width_pix,
                                                img_width=0.8,
                                                n_detectors=n_detectors,
                                                n_views=n_views,
                                                materials=["bone", "water", "iodine"],
                                                img_width_unit='fov_frac',
                                                device=self.device,
                                                settings_data_path=detector_settings_path)

        self.fwd_acc, self.bwd_acc = assemble_fwd_bwd_modules(components, psf_width=21)
        self.fwd_apx, self.bwd_apx = assemble_fwd_bwd_modules(components, psf_width=1)

        rel_params = np.array([1, 1, 1])#np.array([2, 4, 1]) #np.array([1, 1, 1]) #
        reg_params = np.array([r**0.5 for r in rel_params/sum(rel_params)])
        self.reg_scale = reg_scale
        self.regulariser, self.regulariser_grad = assemble_huber_regulariser(components,
                                                                             reg_params=reg_params,
                                                                             huber_param=[0.05, 0.05, 0.05]# 0.07, 0.05, 0.06]#0.01,0.001) # 0.05,0.05,0.03
                                                                             )
        self.air_sino = self.fwd_apx(torch.zeros(1, 3, img_width_pix, img_width_pix))
        #self.normalisation = self.fwd_apx(torch.zeros(1, 3, img_width_pix, img_width_pix)).max() * 10.
        self.normalisation = self.air_sino.max() * 10.
        self.bwd_normalisation = 1e-5


        self.phantom_cell_area = components["image_space"].weighting.const
        self.sinogram_cell_area = components["sinogram_space"].weighting.const
        self.bfgs_stepsize = bfgs_stepsize
        self.bfgs_num_store = 10


        self.print("Creating corrective nets...")

        # Forward correction
        self.epoch = 0
        self.fwd_corr_net = ResUnet(in_channels=8,
                                    min_out_channels=in_ch[0],
                                    depth=4,
                                    scale=None,
                                    batch_norm=batch_norm[0],
                                    activation=activations[0]).to(self.device)

        #self.fwd_corr_net = nn.Conv2d(in_channels=8, out_channels=8, padding_mode="circular", kernel_size=1, stride=1, padding=0).to(self.device)

        self.fwd_loss_fn = nn.MSELoss().to(self.device)
        self.fwd_optimizer = torch.optim.Adam(self.fwd_corr_net.parameters(), lr=learning_rate[0])
        #self.fwd_optimizer = torch.optim.SGD(self.fwd_corr_net.parameters(), lr=learning_rate[0], momentum=0.5)

        # Backward correction
        self.bwd_corr_net = ResUnet(in_channels=3,
                                    min_out_channels=in_ch[1],
                                    depth=4,
                                    scale=None,
                                    padding_mode="reflect",
                                    batch_norm=batch_norm[1],
                                    activation=activations[1]).to(self.device)  # 32 before # zeros before

        #self.bwd_corr_net = nn.Conv2d(in_channels=3, out_channels=3, padding_mode="circular", kernel_size=1, stride=1, padding=0).to(self.device)

        self.bwd_loss_fn = nn.MSELoss().to(self.device)
        self.bwd_optimizer = torch.optim.Adam(self.bwd_corr_net.parameters(), lr=learning_rate[1])
        #self.bwd_optimizer = torch.optim.SGD(self.bwd_corr_net.parameters(), lr=learning_rate[1], momentum=0.5)
        self.grad_clip = grad_clip

        # Load data!
        self.print("Preparing data ...")

        self.num_data = num_data
        self.dataset = TvDataSet(path=self.session_data_iter_path, verbose=self.verbose, device=self.device)

        # If this is a new session, create a folder for storing data. Preferrably an SSD.
        if not is_old_session:
            self.copy_init_data()
            with open(self.log_path, mode='a+') as f:
                # f.write("epoch,batch,phantom_id,phantom_iter,loss_fwd,loss_bwd,val_fwd,val_bwd\n")
                f.write(
                    "epoch,batch,phantom_id,phantom_iter,loss_fwd,loss_bwd,val_fwd,val_bwd,bench_fwd,bench_bwd,bwd_align,fwd_grad_norm,bwd_grad_norm\n")
            torch.save(self.static_settings, self.session_path + "static_settings.dict")
            torch.save(torch.load(detector_settings_path), self.session_path + "detector_settings.dict")
            self.save_state()
        # If this is an old session, load the latest checkpoint.
        else:
            # Determine current epoch
            logs = pd.read_csv(self.log_path)
            if len(logs.index) == 0:
                self.epoch = 0
            else:
                self.epoch = logs["epoch"].max()
            # Find the newest save state.
            sessions = os.listdir(self.session_states_path)
            for i in range(self.epoch, 0, -1):
                fwd_path = f"fwd_corr_epoch_{i}.StateDict"
                bwd_path = f"bwd_corr_epoch_{i}.StateDict"
                if (fwd_path in sessions) and (bwd_path in sessions):
                    print(f"Loading closest checkpoint, {i}")
                    self.load_state(self.session_states_path + fwd_path,
                                    self.session_states_path + bwd_path)
                    break
            if self.num_data is None:
                self.num_data = len(self.dataset)

        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=1, shuffle=True)

        # Session logs
        self.logger = EventTracker()
        self.validate_every_n = validate_every_n
        self.save_every_n = save_every_n

    # EVALUATING
    def data_fidelity(self, y, y_true):
        y_log_yt_over_y = torch.where(y_true == 0, torch.zeros_like(y), y_true * torch.log(y_true / y))
        return self.sinogram_inner(y - y_true + y_log_yt_over_y, torch.ones_like(y_true)) / self.normalisation

    def data_fidelity_sinogram_gradient(self, y, y_true):
        return (1 - y_true / y) * self.sinogram_cell_area / self.normalisation

    def data_fidelity_phantom_gradient(self, x, f, fadj, y_true):
        return fadj([x, self.data_fidelity_sinogram_gradient(f(x), y_true)])

    def loss(self, x, f, y_true):
        return self.data_fidelity(f(x), y_true) + self.reg_scale * self.regulariser(x)

    def loss_gradient(self, x, f, f_adj, y_true):
        return self.data_fidelity_phantom_gradient(x, f, f_adj, y_true) + self.reg_scale * self.regulariser_grad(x)

    def loss_gradient_autograd(self, x, fwd, y_true):
        x.grad = None
        x.requires_grad_(True)
        loss = self.data_fidelity(fwd(x), y_true)
        loss.backward()
        xgrad = x.grad
        x.requires_grad_(False)
        return xgrad + self.reg_scale * self.regulariser_grad(x)


    def phantom_inner(self, x, y):
        return (torch.tensordot(x, y, dims=x.dim()) * self.phantom_cell_area)

    def sinogram_inner(self, x, y):
        return (torch.tensordot(x, y, dims=x.dim()) * self.sinogram_cell_area)

    def fwd_correction(self, x):
        return self.fwd_corr_net(self.fwd_apx(x) / self.air_sino) * self.air_sino

    def bwd_correction(self, xy):
        return self.bwd_corr_net(self.bwd_apx(xy) / self.bwd_normalisation) * self.bwd_normalisation

    # LOADING AND SAVING
    def get_y_true(self, idx):
        return self.load_to_device(self.data_target_path + f"target_sinogram_{idx}.Tensor")

    def get_x_true(self, idx):
        return self.load_to_device(self.data_target_path + f"target_phantom_{idx}.Tensor")

    def load_to_device(self, path):
        return torch.load(path).to(self.device)

    def copy_init_data(self):
        files_in_init = os.listdir(self.data_init_path)
        # Loop through data folder,
        for idx in range(self.num_data):
            filename = f"phantom_{idx}.Tensor"
            assert filename in files_in_init, "Not enough data in this folder."
            # Give information
            self.print(f"[{idx + 1}/{self.num_data}] Datapoints loaded", end="\r")
            # Load data and save new data to file.
            x = self.load_to_device(self.data_init_path + filename)
            y_true = self.get_y_true(idx)
            x_data = DataPoint.dataPoint_from_ops(x=x,
                                                  fwd_accurate=self.fwd_acc,
                                                  fwd_approx=self.fwd_apx,
                                                  fwd_adj_acc=self.bwd_acc,
                                                  likelihood_grad=self.data_fidelity_sinogram_gradient,
                                                  y_true=y_true,
                                                  id=idx,
                                                  iter=0)
            self.dataset.append(x_data)

    def makedirs(self):
        try:
            os.makedirs(self.session_path, exist_ok=False)
            os.makedirs(self.session_data_iter_path, exist_ok=False)
            os.makedirs(self.session_states_path, exist_ok=False)
            os.makedirs(self.script_path, exist_ok=False)
            os.makedirs(self.session_path + "output_img/", exist_ok=False)
            already_exists = False
        except:
            already_exists = True
        if already_exists:
            self.print("Directory already exists! Loading session ...")
        else:
            self.print("Successfully created directories:")
            self.print(self.session_path)
            self.print(self.session_data_iter_path)
            self.print(self.session_states_path)
            self.print(self.script_path)
        return already_exists

    def print(self, msg, **kwargs):
        if self.verbose:
            print(msg, **kwargs)
        return 0

    def save_state(self):
        self.print("Saving current state dicts.", end="\r")
        # Save network parameters
        torch.save(self.fwd_corr_net.state_dict(), self.session_states_path + f"fwd_corr_epoch_{self.epoch}.StateDict")
        torch.save(self.bwd_corr_net.state_dict(), self.session_states_path + f"bwd_corr_epoch_{self.epoch}.StateDict")
        # Save input settings
        torch.save(self.dynamic_settings, self.session_states_path + f"dynamic_settings_{self.epoch}.dict")

        self.print("Saved current state dicts.")
        return 0

    def load_state(self, fwd_path=None, bwd_path=None):
        self.print("Loading states...", end="\r")
        if fwd_path is not None:
            self.fwd_corr_net.load_state_dict(torch.load(fwd_path))
        if bwd_path is not None:
            self.bwd_corr_net.load_state_dict(torch.load(bwd_path))
        self.print("Done loading states.")

    def load_state_epoch(self, epoch):
        self.load_state(self.session_states_path + f"fwd_corr_epoch_{epoch}.StateDict",
                        self.session_states_path + f"bwd_corr_epoch_{epoch}.StateDict")

    def get_settings(self):
        return self.static_settings, self.dynamic_settings

    # TRAINING
    def validate(self):
        """
        Validate on separate data.
        This operation is expensive, so only do once in a while
        :param n_avg:
        :return:
        """
        # Change mode
        self.eval_mode()

        # Evaluate losses
        with torch.no_grad():
            # Load data
            file = np.random.choice(os.listdir(self.data_test_path))
            x_val = torch.load(self.data_test_path + file).to(self.device)

            # Compute forward corrections
            y_acc = self.fwd_acc(x_val)
            y_true = torch.poisson(y_acc)
            y_apx = self.fwd_apx(x_val)

            # Normalise data
            y_apx_norm = y_apx / self.air_sino
            y_acc_norm = y_acc / self.air_sino
            y_corr_norm = self.fwd_corr_net(y_apx_norm)
            y_corr = self.air_sino * y_corr_norm

            # Compute backward corrections
            yadj_acc = self.bwd_acc([x_val, self.data_fidelity_sinogram_gradient(y_acc, y_true)])
            yadj_accbwd = self.bwd_acc([x_val, self.data_fidelity_sinogram_gradient(y_corr, y_true)])
            yadj_corr = self.bwd_corr_net(self.bwd_apx([x_val, self.data_fidelity_sinogram_gradient(y_corr, y_true)])/self.bwd_normalisation)*self.bwd_normalisation

            # Compute errors
            inner = self.phantom_inner
            bwd_val = self.bwd_loss_fn(yadj_corr, yadj_accbwd).item() / self.bwd_normalisation ** 2 #### NEW HOTFIX
            bwd_align = (1 - inner(yadj_corr, yadj_acc) / (inner(yadj_corr, yadj_corr) * inner(yadj_acc, yadj_acc)) ** 0.5).item()
            # fwd_val = self.fwd_loss_fn(y_corr_norm/y_acc_norm, torch.ones_like(y_acc)).item()
            fwd_val = self.fwd_loss_fn(y_corr_norm, y_acc_norm).item()

        # Change mode
        self.training_mode()
        return fwd_val, bwd_val, bwd_align

    def eval_mode(self):
        self.fwd_corr_net.eval()
        self.bwd_corr_net.eval()

    def training_mode(self):
        self.fwd_corr_net.train()
        self.bwd_corr_net.train()

    def train_step(self, x, y_acc, y_apx, yadj_acc, y_true, train_forward=True, train_backward=True, callback=None):
        """
        Train the network on a given datapoint.
        :param data:
        :return:
        """

        # Obtain normalised data!
        self.logger.start_event("Normalising data")
        y_apx_norm = y_apx / self.air_sino  #/self.normalisation
        y_acc_norm = y_acc / self.air_sino  #/self.normalisation
        self.logger.end_event("Normalising data")

        # Forward step!
        if train_forward:
            self.logger.start_event("Forward correction step")
            self.fwd_optimizer.zero_grad()
            y_corr_norm = self.fwd_corr_net(y_apx_norm)
            # fwd_loss = self.fwd_loss_fn(y_corr_norm / y_acc_norm, torch.ones_like(y_acc_norm))
            fwd_loss = self.fwd_loss_fn(y_corr_norm, y_acc_norm)
            fwd_loss.backward()
            if self.grad_clip[0] is not None:
                nn.utils.clip_grad_norm_(self.fwd_corr_net.parameters(), max_norm=self.grad_clip[0])
            self.fwd_optimizer.step()
            self.logger.end_event("Forward correction step")
            fwd_loss = fwd_loss.item()
        else:
            fwd_loss = None

        # Backward step!
        self.logger.start_event("Backward correction step")
        y_corr = self.air_sino * self.fwd_corr_net(y_apx_norm)
        like_grad_cor = self.data_fidelity_sinogram_gradient(y_corr, y_true)
        bwd_apx = self.bwd_apx([x, like_grad_cor])  # INCORRECT
        bwd_acc = self.bwd_acc([x, like_grad_cor])  # Remove if we train on the true direction instead

        self.bwd_optimizer.zero_grad()
        if train_backward:
            bwd_corr = self.bwd_corr_net(bwd_apx / self.bwd_normalisation) * self.bwd_normalisation  # HOTFIX
            bwd_loss = self.bwd_loss_fn(bwd_corr, bwd_acc) / self.bwd_normalisation ** 2 # HOTFIX 2
            bwd_loss.backward()
            if self.grad_clip[1] is not None:
                nn.utils.clip_grad_norm_(self.bwd_corr_net.parameters(), max_norm=self.grad_clip[1])
            self.bwd_optimizer.step()
            bwd_loss = bwd_loss.item()
            # print(f"Gradient loss {self.phantom_inner(bwd_corr, yadj_acc) / self.phantom_inner(yadj_acc, yadj_acc)}," +
            #      f" Aim for {self.phantom_inner(bwd_apx, yadj_acc) / self.phantom_inner(yadj_acc, yadj_acc)}")
        else:
            bwd_corr = None
            bwd_loss = None


        self.logger.end_event("Backward correction step")

        # Benchmark losses
        self.logger.start_event("Benchmark data")
        # apx_loss_fwd = self.fwd_loss_fn(y_apx_norm / y_acc_norm, torch.ones_like(y_acc)).item()
        apx_loss_fwd = self.fwd_loss_fn(y_apx_norm, y_acc_norm).item()
        apx_loss_bwd = self.bwd_loss_fn(bwd_apx, bwd_acc).item() / self.bwd_normalisation ** 2
        self.logger.end_event("Benchmark data")

        if callback is not None:
            callback(x=x,
                     y_acc=y_acc,
                     y_apx=y_apx,
                     y_corr=y_corr,
                     bwd_acc=bwd_acc,
                     bwd_apx=bwd_apx,
                     bwd_corr=bwd_corr)

        return fwd_loss, bwd_loss, apx_loss_fwd, apx_loss_bwd

    # Regular training
    def run_epoch(self, epoch, forward=True, backward=True, callback=None):
        logs = {"epoch": [], "batch": [], "data_id": [], "data_iter": [],
                "loss_fwd": [], "loss_bwd": [], "val_fwd": [], "val_bwd": [],
                "bench_fwd": [], "bench_bwd": [], "val_align": [], "fwd_grad_norm":[], "bwd_grad_norm": []}

        # Quick hack to shuffle data
        data_indices = list(range(self.num_data))
        np.random.shuffle(data_indices)

        # Iterate through the data
        for batch, data_idx in enumerate(data_indices):

            self.logger.start_event("Loading_data_point")
            data = self.dataset.__getitem__(data_idx)
            self.logger.end_event("Loading_data_point")

            # Load true sinogram
            self.logger.start_event("Loading true data")
            y_true = self.get_y_true(data.id)
            self.logger.end_event("Loading true data")

            # Training step
            self.logger.start_event("Training step")
            fwd_loss, bwd_loss, apx_fwd_loss, apx_bwd_loss = self.train_step(x=data.x,
                                                                             y_apx=data.y_apx,
                                                                             y_acc=data.y_acc,
                                                                             yadj_acc=data.yadj_acc,
                                                                             y_true=y_true,
                                                                             train_forward=forward,
                                                                             train_backward=backward,
                                                                             callback=callback)
            self.logger.end_event("Training step")

            logs["epoch"].append(epoch)
            logs["batch"].append(batch)
            logs["data_id"].append(data.id)
            logs["data_iter"].append(data.iter)
            logs["loss_fwd"].append(fwd_loss)
            logs["loss_bwd"].append(bwd_loss)
            logs["bench_fwd"].append(apx_fwd_loss)
            logs["bench_bwd"].append(apx_bwd_loss)
            logs["fwd_grad_norm"].append((sum([p.grad.norm().item() ** 2 for p in self.fwd_corr_net.parameters() if p.grad is not None])**0.5))
            logs["bwd_grad_norm"].append((sum([p.grad.norm().item() ** 2 for p in self.bwd_corr_net.parameters() if p.grad is not None]) ** 0.5))


            # If allowed, validate!
            if batch % self.validate_every_n == 0:
                self.logger.start_event("Validation")
                val_fwd, val_bwd, val_align = self.validate()
                self.logger.end_event("Validation")
                logs["val_fwd"].append(val_fwd)
                logs["val_bwd"].append(val_bwd)
                logs["val_align"].append(val_align)
            else:
                logs["val_fwd"].append(None)
                logs["val_bwd"].append(None)
                logs["val_align"].append(None)
            self.print(f"Batch [{batch + 1}/{len(self.dataloader)}], " +
                       f"phantom {data.id}, iter {data.iter}, " +
                       f"fwd loss {fwd_loss} - aim for {apx_fwd_loss}, " +
                       f"bwd loss {bwd_loss} - aim for {apx_bwd_loss}", end="\r")
        return logs

    # Recursive
    def run_epoch_recursion(self, epoch, recursions: int = 0, train_every_n: int = 100, save_every_n_steps=1000, callback=None):
        logs = {"epoch": [], "batch": [], "data_id": [], "data_iter": [],
                "loss_fwd": [], "loss_bwd": [], "val_fwd": [], "val_bwd": [],
                "bench_fwd": [], "bench_bwd": [], "val_align": [], "fwd_grad_norm": [], "bwd_grad_norm": []}
        # Quick hack to shuffle data
        # np.random.seed(0)
        data_indices = list(range(self.num_data))
        np.random.shuffle(data_indices)

        # Iterate through the data
        for batch, data_idx in enumerate(data_indices):

            self.logger.start_event("Loading_data_point")
            data = self.dataset.__getitem__(data_idx)
            self.logger.end_event("Loading_data_point")

            # Load true sinogram
            self.logger.start_event("Loading true data")
            y_true = self.get_y_true(data.id)
            self.logger.end_event("Loading true data")

            # We don't care about the computation graph here, only in every tenth step or so
            # Correctly computed gradient
            def grad(x):
                with torch.no_grad():
                    return self.loss_gradient(x, self.fwd_correction, self.bwd_correction, y_true)

            def train_callback(x, n_recurse, ys, ss):
                TOL = 100  # TOLERANCE TO STOP RECURSIONS

                if (batch * recursions + n_recurse) % save_every_n_steps == 0:
                    self.logger.start_event("saving over epoch with new state (Hotfix for recursive training)")
                    self.save_state()
                    self.logger.end_event("saving over epoch with new state (Hotfix for recursive training)")

                if n_recurse % train_every_n == 0:
                    if not x.isnan().any() or torch.abs(x).max() > TOL:
                        self.logger.start_event("Saving image")
                        if n_recurse % 50 == 0:
                            torch.save(x.clone(), self.session_path + f"output_img/phantom_{n_recurse}.Tensor")
                        self.logger.end_event("Saving image")

                        self.logger.start_event("Generating data")
                        y_apx = self.fwd_apx(x)
                        y_acc = self.fwd_acc(x)

                        # Incorrectly computed yadj before, skipped the weighting
                        yadj_acc = self.bwd_acc([x, self.data_fidelity_sinogram_gradient(y_acc, y_true)])
                        self.logger.end_event("Generating data")

                        self.logger.start_event("Training")
                        self.training_mode()
                        fwd_loss, bwd_loss, apx_loss_fwd, apx_loss_bwd = self.train_step(x=x,
                                                                                         y_acc=y_acc,
                                                                                         y_apx=y_apx,
                                                                                         yadj_acc=yadj_acc,
                                                                                         y_true=y_true,
                                                                                         train_forward=True,
                                                                                         train_backward=True,
                                                                                         callback=callback)
                        self.logger.end_event("Training")

                        logs["epoch"].append(epoch)
                        logs["batch"].append(batch)
                        logs["data_id"].append(data.id)
                        logs["data_iter"].append(n_recurse)
                        logs["loss_fwd"].append(fwd_loss)
                        logs["loss_bwd"].append(bwd_loss)
                        logs["bench_fwd"].append(apx_loss_fwd)
                        logs["bench_bwd"].append(apx_loss_bwd)
                        logs["fwd_grad_norm"].append((sum(
                            [p.grad.norm().item() ** 2 for p in self.fwd_corr_net.parameters() if
                             p.grad is not None]) ** 0.5))
                        logs["bwd_grad_norm"].append((sum(
                            [p.grad.norm().item() ** 2 for p in self.bwd_corr_net.parameters() if
                             p.grad is not None]) ** 0.5))

                        # If allowed, validate!
                        if batch % self.validate_every_n == 0:
                            self.logger.start_event("Validation")
                            val_fwd, val_bwd, val_align = self.validate()
                            self.logger.end_event("Validation")
                            logs["val_fwd"].append(val_fwd)
                            logs["val_bwd"].append(val_bwd)
                            logs["val_align"].append(val_align)
                        else:
                            logs["val_fwd"].append(None)
                            logs["val_bwd"].append(None)
                            logs["val_align"].append(None)

                        self.print(f"Batch [{batch + 1}/{len(self.dataloader)}], " +
                                   f"Recursion [{n_recurse} / {recursions}]" +
                                   f"phantom {data.id}, iter {data.iter}, " +
                                   f"fwd loss {fwd_loss} - aim for {apx_loss_fwd}, " +
                                   f"bwd loss {bwd_loss} - aim for {apx_loss_bwd}")  # , end="\r")
                        return 0
                    else:
                        self.print(f"Exploding gradient at recursion {n_recurse}!")
                        return -1


            self.logger.start_event("Full Recursion")
            self.eval_mode()
            self.bfgs_steps(x=data.x,
                            grad=grad,
                            inner=self.phantom_inner,
                            iter=recursions,
                            callback=train_callback,
                            step=self.bfgs_stepsize,
                            num_store=self.bfgs_num_store,
                            logger=self.logger)
            self.logger.end_event("Full Recursion")
        return logs

    # Minibatches
    def run_epoch_minibatches(self, epoch, batch_size, forward=True, backward=True, callback=None):
        logs = {"epoch": [], "batch": [], "data_id": [], "data_iter": [],
                "loss_fwd": [], "loss_bwd": [], "val_fwd": [], "val_bwd": [],
                "bench_fwd": [], "bench_bwd": [], "val_align": [], "fwd_grad_norm": [], "bwd_grad_norm": []}

        data_indices = list(range(self.num_data))
        np.random.shuffle(data_indices)
        batches = np.split(np.array(data_indices), range(batch_size, len(data_indices), batch_size))

        # Iterate through the data
        for batch_id, batch in enumerate(batches):
            # batch_id = 0
            # batch = batches[0]

            x_tensor = torch.zeros((len(batch), 3, 128, 128), device=self.device)
            y_apx_tensor = torch.zeros((len(batch), 8, 128, 128), device=self.device)
            y_acc_tensor = torch.zeros((len(batch), 8, 128, 128), device=self.device)
            y_true_tensor = torch.zeros((len(batch), 8, 128, 128), device=self.device)

            for i, idx in enumerate(batch):
                self.logger.start_event("Loading_data_point")
                data = self.dataset.__getitem__(idx)
                self.logger.end_event("Loading_data_point")

                # Load true sinogram
                self.logger.start_event("Loading true data")
                y_true = self.get_y_true(data.id)
                self.logger.end_event("Loading true data")

                x_tensor[i] = data.x
                y_apx_tensor[i] = data.y_apx
                y_acc_tensor[i] = data.y_acc
                y_true_tensor[i] = y_true

            # Training step
            self.logger.start_event("Training step")
            fwd_loss, bwd_loss, apx_fwd_loss, apx_bwd_loss = self.train_step(x=x_tensor,
                                                                             y_apx=y_apx_tensor,
                                                                             y_acc=y_acc_tensor,
                                                                             yadj_acc=None,
                                                                             y_true=y_true_tensor,
                                                                             train_forward=forward,
                                                                             train_backward=backward,
                                                                             callback=callback)

            self.logger.end_event("Training step")
            logs["epoch"].append(epoch)
            logs["batch"].append(batch_id)
            logs["data_id"].append(-1)
            logs["data_iter"].append(0)
            logs["loss_fwd"].append(fwd_loss)
            logs["loss_bwd"].append(bwd_loss)
            logs["bench_fwd"].append(apx_fwd_loss)
            logs["bench_bwd"].append(apx_bwd_loss)
            logs["fwd_grad_norm"].append(
                (sum([p.grad.norm().item() ** 2 for p in self.fwd_corr_net.parameters() if p.grad is not None]) ** 0.5))
            logs["bwd_grad_norm"].append(
                (sum([p.grad.norm().item() ** 2 for p in self.bwd_corr_net.parameters() if p.grad is not None]) ** 0.5))

            # If allowed, validate!
            if batch_id % self.validate_every_n == 0:
                self.logger.start_event("Validation")
                val_fwd, val_bwd, val_align = self.validate()
                self.logger.end_event("Validation")
                logs["val_fwd"].append(val_fwd)
                logs["val_bwd"].append(val_bwd)
                logs["val_align"].append(val_align)
            else:
                logs["val_fwd"].append(None)
                logs["val_bwd"].append(None)
                logs["val_align"].append(None)
            self.print(f"Batch [{batch_id + 1}/{len(batches)}], " +
                       f"size: {len(batch)}, iter {0}, " +
                       f"fwd loss {fwd_loss} - aim for {apx_fwd_loss}, " +
                       f"bwd loss {bwd_loss} - aim for {apx_bwd_loss}", end="\r")
        return logs

    # Multi-epoch versions
    def run_epochs(self, n_epochs, forward=True, backward=True, callback=None):
        np.random.seed(0)
        for n in range(n_epochs):
            self.logger.start_event("epoch")
            self.epoch += 1
            logs = self.run_epoch(self.epoch, forward=forward, backward=backward, callback=callback)
            with open(self.log_path, mode='a+') as f:
                for i in range(len(logs["epoch"])):
                    f.write(','.join([str(logs[k][i]) for k in logs.keys()]) + "\n")
            self.logger.end_event("epoch")

            if n % self.save_every_n == 0:
                self.logger.start_event("saving epoch")
                self.save_state()
                self.logger.end_event("saving epoch")


            self.print(f"[{n + 1}/{n_epochs} Epochs done.] Total epochs: {self.epoch}")

    def run_epochs_recursion(self, recursions, train_every_n, save_every_n_steps, callback=None):
        np.random.seed(0)
        n_epochs = len(recursions)
        for n in range(n_epochs):
            self.logger.start_event("epoch")
            self.epoch += 1
            logs = self.run_epoch_recursion(self.epoch, recursions[n], train_every_n, save_every_n_steps, callback)
            with open(self.log_path, mode='a+') as f:
                for i in range(len(logs["epoch"])):
                    f.write(','.join([str(logs[k][i]) for k in logs.keys()]) + "\n")
            self.logger.end_event("epoch")

            if n % self.save_every_n == 0:
                self.logger.start_event("saving epoch")
                self.save_state()
                self.logger.end_event("saving epoch")

            self.print(f"[{n + 1}/{n_epochs} Epochs done.] Total epochs: {self.epoch}")

    def run_epochs_minibatches(self, n_epochs, batch_size, forward=True, backward=True, callback=None):
        for n in range(n_epochs):
            self.logger.start_event("epoch")
            self.epoch += 1
            logs = self.run_epoch_minibatches(self.epoch, batch_size, forward, backward, callback)
            with open(self.log_path, mode='a+') as f:
                for i in range(len(logs["epoch"])):
                    f.write(','.join([str(logs[k][i]) for k in logs.keys()]) + "\n")
            self.logger.end_event("epoch")

            if n % self.save_every_n == 0:
                self.logger.start_event("saving epoch")
                self.save_state()
                self.logger.end_event("saving epoch")

            self.print(f"[{n + 1}/{n_epochs} Epochs done.] Total epochs: {self.epoch}")

    def get_saving_callback(self, every_n, capacity, path):
        class CallBack(odl.solvers.Callback):
            def __init__(self):
                super(CallBack, self).__init__()
                self.iter = 0
                self.current_memory_pos = 0
                self.capacity = capacity
                self.every_n = every_n
                self.path = path

            def __call__(self, **kwargs):
                if self.iter % self.every_n == 0:
                    torch.save(kwargs, self.path + f"state_{self.current_memory_pos}.dict")
                    self.current_memory_pos = (self.current_memory_pos + 1) % self.capacity
                self.iter += 1

        return CallBack()

    def get_log(self):
        return self.log_from_path(self.log_path)
    
    def to_device(self, device):
        self.fwd_corr_net.to(device)
        self.bwd_corr_net.to(device)
        
        #self.fwd_apx.to(device)
        #self.bwd_apx.to(device)
        
        #self.fwd_acc.to(device)
        #self.bwd_acc.to(device)
        
    @staticmethod
    def log_from_path(path):
        log = pd.read_csv(path)
        log = log.mask(log.eq("None")).astype(float)
        return log

    @staticmethod
    def load_old_session(name, path, epoch=None, **settings):
        if epoch is None:
            try:
                with open(path + name + "/session_log.txt", 'rb') as f:
                    try:
                        f.seek(-2, os.SEEK_END)
                        while f.read(1) != b'\n':
                            f.seek(-2, os.SEEK_CUR)
                    except OSError:
                        f.seek(0)
                    epoch = int(str.split(f.readline().decode(), ",")[0])
                while epoch >= 0 and not os.path.isfile(path + name + f"/session_states/dynamic_settings_{epoch}.dict"):
                    epoch -= 1
                print(f"Picking most recent save: {epoch}")
                ses = FwdBwdSession.load_old_session(name, path, epoch=epoch, **settings)
            except:
                assert False, "Session corrupted."
        else:
            try:
                static_settings = torch.load(path + name + "/static_settings.dict")
                dynamic_settings = torch.load(path + name + f"/session_states/dynamic_settings_{epoch}.dict")
            except:
                assert False, "Session or requested epoch does not exist."

            for setting in settings.keys():
                dynamic_settings[setting] = settings[setting]

            ses = FwdBwdSession(**static_settings, **dynamic_settings)
            ses.load_state_epoch(epoch)
        return ses

    @staticmethod
    def bfgs_steps(x, grad, inner, step: float = 1.0, iter: int = 1000, tol=1e-15,
                   num_store=None, hessinv_estimate=None, callback=None, logger=None, verbose=False):
        r"""Quasi-Newton BFGS method to minimize a differentiable function.
        For minimising f given its gradient and initial point x. Also needs the inner product, which depends on the space.
        """

        def start_event(event):
            if logger is not None:
                logger.start_event(event)
            return

        def end_event(event):
            if logger is not None:
                logger.end_event(event)
            return

        ys = []
        ss = []
        grad_x = grad(x)
        # plt.imshow(grad_x.cpu().squeeze()[1])

        for i in range(iter):
            if verbose:
                if logger is None:
                    print(f"{i}/{iter} iterations", end="\r")
                else:
                    print(f"{i}/{iter} iterations, log: " + str(logger), end="\r")
                        
            # Determine a stepsize using line search
            start_event("bfgs direction")
            search_dir = -FwdBwdSession.bfgs_direction_torch(ss, ys, grad_x, inner, hessinv_estimate)
            # plt.imshow(search_dir.cpu().squeeze()[2])
            # plt.imshow(x.cpu().squeeze()[2])
            # plt.imshow(grad_diff.cpu().squeeze()[2])
            # plt.imshow(grad_x.cpu().squeeze()[2])
            # plt.colorbar()
            end_event("bfgs direction")

            dir_deriv = inner(search_dir, grad_x).item()
            if np.abs(dir_deriv) == 0:
                return  # we found an optimum

            # Update x
            start_event("update")
            x_update = search_dir
            x_update = step * x_update
            x = x + x_update
            end_event("update")

            start_event("gradient")
            grad_x, grad_diff = grad(x), grad_x
            # grad_diff = grad(x) - grad(x_old)
            grad_diff = grad_x - grad_diff
            end_event("gradient")

            y_inner_s = inner(grad_diff, x_update).item()

            # Test for convergence
            if np.abs(y_inner_s) < tol:
                if verbose:
                    print(
                        f"y_inner_s too small: {np.abs(y_inner_s)}, and gradient norm: {inner(grad_x, grad_x)}, Resetting memory.",
                        end="\r")
                if inner(grad_x, grad_x) < tol:
                    if verbose:
                        print(f"Converged after {i} steps.")
                    return x, ys, ss
                else:
                    # Reset if needed
                    ys = []
                    ss = []
                    continue

            # Update Hessian
            ys.append(grad_diff)
            ss.append(x_update)
            if num_store is not None:
                # Throw away factors if they are too many.
                if num_store > 0:
                    ss = ss[-num_store:]
                    ys = ys[-num_store:]
                else:
                    ss = []
                    ys = []
            if callback is not None:
                start_event("callback")
                state = callback(x, i, ys, ss)
                end_event("callback")
                if state == -1:
                    break

        return x, ys, ss

    @staticmethod
    def bfgs_direction_torch(s, y, x, inner, hessinv_estimate=None):
        r"""Compute ``Hn^-1(x)`` for the L-BFGS method.

        Parameters
        ----------
        s : sequence of `LinearSpaceElement`
            The ``s`` coefficients in the BFGS update, see Notes.
        y : sequence of `LinearSpaceElement`
            The ``y`` coefficients in the BFGS update, see Notes.
        x : `LinearSpaceElement`
            Point in which to evaluate the product.
        hessinv_estimate : `Operator`, optional
            Initial estimate of the hessian ``H0^-1``.

        Returns
        -------
        r : tensor element
            The result of ``Hn^-1(x)``.
        """
        assert len(s) == len(y)

        r = x.clone()
        alphas = torch.zeros(len(s))
        rhos = torch.zeros(len(s))

        for i in reversed(range(len(s))):
            rhos[i] = 1.0 / inner(y[i], s[i])
            alphas[i] = rhos[i] * inner(s[i], r)
            r -= alphas[i] * y[i]

        if hessinv_estimate is not None:
            r = hessinv_estimate(r)

        for i in range(len(s)):
            beta = rhos[i] * inner(y[i], r)
            r += (alphas[i] - beta) * s[i]
        return r

    @staticmethod
    def sqs_steps(x, hess, grad, inner, iter, nesterov=False, callback=None, logger=None, tol=1e-15, verbose=False):

        def start_event(event):
            if logger is not None:
                logger.start_event(event)
            return

        def end_event(event):
            if logger is not None:
                logger.end_event(event)
            return

        # Loss function: L(x' + dx) = L(x') + grad(L,x') * dx + 1/2 dx*hess(L,x')*dx
        # To minimise then, is to solve hess(L,x')dx = grad(L,x').
        # Therefore, step size will be dx = inv(hess(L,x'))grad(L,x').
        # If x has dimension (1,3,128,128), then hess has dimension (1,3,3,128,128)  -> all channel cross terms
        grad_x = grad(x)
        hess_x = hess(x)
        # Calculate step
        dx = torch.linalg.torch.solve(grad_x.permute((0, 2, 3, 1))[:, :, :, :, None],
                                      hess_x.permute((0, 3, 4, 1, 2)),
                                      ).permute((0, 3, 1, 2, 4)).squeeze(dim=4)
        # Take step!
        x = x+dx

        # plt.imshow(grad_x.cpu().squeeze()[1])

        for i in range(iter):
            # Determine a stepsize using line search
            start_event("sqs stepping")
            # Gradient, hessian approximate
            grad_x = grad(x)
            hess_x = hess(x)
            # Calculate step
            dx = torch.linalg.torch.solve(-grad_x.permute((0, 2, 3, 1))[:, :, :, :, None],
                                          hess_x.permute((0, 3, 4, 1, 2)),
                                          ).permute((0, 3, 1, 2, 4)).squeeze(dim=4)
            # Take step!
            x = x + dx
            search_dir = 0
            end_event("sqs stepping")


            # Update x
            start_event("Nesterov acceleration")
            #
            end_event("Nesterov acceleration")


            if inner(grad_x, grad_x) < tol:
                if verbose:
                    print(f"Converged after {i} steps.")
                return x

            # Callback (save, print etc.)
            if callback is not None:
                start_event("callback")
                state = callback(x, i)
                end_event("callback")
                if state == -1:
                    break

        return x

    @staticmethod
    def sqs_step(x, ytrue, reg_par):
        """
        One step of separable quadratic surrogates.
        The most general version of this method uses a subsample of the rays for projection.
        There doesn't seem to be an easy way of doing this in odl, so we choose 1 batch.
        :return:
        """

        return x

    @staticmethod
    def runs():
        ses = FwdBwdSession(session_name="2021-08-04_recursive_8ch_clip_norm",
                            data_loading_path="C:/Fastprojects/sessions_summer/",
                            session_path="data/sessions_summer/",
                            learning_rate=[1e-3, 1e-3],
                            num_data=100,
                            device="cuda",
                            verbose=True,
                            validate_every_n=100,
                            save_every_n=10,
                            batch_norm=(0, 0),
                            activations=(nn.LeakyReLU, nn.LeakyReLU),
                            in_ch=(8, 8),
                            bfgs_stepsize=1.0,
                            grad_clip=(0.002, 0.002),
                            reg_scale=0.000006)

        ses = FwdBwdSession.load_old_session("2021-07-16_operator_tests", "data/sessions_summer/", epoch=0, bfgs_stepsize=0.8)

        x = ses.dataset.__getitem__(0).x
        plot_image_channels([(ses.fwd_apx(x)/ses.fwd_acc(x)-1).cpu()], subset = [0, 1, 3, 7], vmin=-0.1, vmax=0.1)
        pass

    @staticmethod
    def check_logs():
        pass
