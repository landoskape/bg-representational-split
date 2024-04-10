# library imports
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
import torchvision


# Define dataset classes
class GratingsDataset(Dataset):
    """
    The gratings dataset is very simple - it uses variable frequency sinusoidal gratings
    with random phase that can either be horizontal or vertical and can take up the whole
    image or only use half the image.

    "N" is arbitrary because a new grating is created every time __getitem__ is used, but is
    required for defining the size of the dataset (e.g. how many times to make gratings before
    completing one "epoch" of this dataset)

    The output is:
    - image: the grating image with shape (channels, D, D)
    - target: the target label for images in this class
    - unique_id: a unique ID associated with this class only
    """

    def __init__(self, freq_range=[1, 5], channels=3, D=32, N=1000, vertical=True, half=False, target=1.0, **kwargs):
        self.D = D
        self.N = N
        self.channels = channels
        self.half = half
        self.vertical = vertical
        self.freq_range = freq_range
        self.get_freq = lambda: torch.diff(torch.tensor(freq_range)) * torch.rand(1) + torch.tensor(freq_range).min()
        self.axis = torch.linspace(0, 2 * torch.pi, D)
        if half:
            self.half_zeros = torch.arange(D).view(1, -1).expand(D, -1) >= D / 2
        self.target = target
        self.unique_id = kwargs.get("unique_id", 2.0 * vertical + 1.0 * half)
        self.name = ("Half" if half else "Full") + " " + ("Vertical" if vertical else "Horizontal")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        c_freq = self.get_freq()
        c_phase = torch.rand(1) * 2 * torch.pi
        data = torch.sin((self.axis - c_phase) * c_freq).view(1, -1).expand(self.D, -1)

        if self.half:
            half = 1.0 * self.half_zeros if torch.rand(1) > 0.5 else 1.0 * (~self.half_zeros)
            data = data * half

        if not self.vertical:
            data = data.T

        data = data.view(1, self.D, self.D).expand(self.channels, -1, -1)
        return data, self.target, self.unique_id


class CifarSingleTarget(torchvision.datasets.CIFAR100):
    """
    The Cifar dataset simply uses the standard test set of CIFAR100, but overwrites
    it to provide a user-defined target and unique ID.

    The output is:
    - image: the cifar image with shape (channels, 32, 32)
    - target: the target label for images in this class
    - unique_id: a unique ID associated with this class only
    """

    def __init__(self, target=0.0, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.unique_id = kwargs.get("unique_id", -1.0)
        self.name = "CIFAR100"

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        return image, self.target, self.unique_id


def MulticlassData(cifar={}, full_vertical={}, half_vertical={}, full_horizontal={}, half_horizontal={}):
    """
    creates dataset object for multi-class datasets
    each variable can contain a few keys which determine how the respective dataset is used
    keys:
    - include: True/False, determines whether to include the dataset
    - target: a number, determines the "target" for that dataset
    -
    datasets should be a list of dictionaries indicating the
    """
    datasets = []

    # cifar100 dataset
    if cifar.get("include", True):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train = cifar.get("train", False)
        target = cifar.get("target", 0)
        cifar = CifarSingleTarget(root="./data", train=train, download=True, transform=transform, target=target)
        datasets.append(cifar)

    # full grating datasets
    if full_vertical.get("include", True):
        target = full_vertical.get("target", 0)
        uid = full_vertical.get("uid", None)
        N = full_vertical.get("N", 2500)
        gratings = GratingsDataset(N=N, vertical=True, half=False, target=target, uid=uid)
        datasets.append(gratings)

    if full_horizontal.get("include", True):
        target = full_horizontal.get("target", 1)
        uid = full_horizontal.get("uid", None)
        N = full_horizontal.get("N", 2500)
        gratings = GratingsDataset(N=N, vertical=False, half=False, target=target, uid=uid)
        datasets.append(gratings)

    # half grating datasets
    if half_vertical.get("include", True):
        target = half_vertical.get("target", 0)
        uid = half_vertical.get("uid", None)
        N = half_vertical.get("N", 2500)
        gratings = GratingsDataset(N=N, vertical=True, half=True, target=target, uid=uid)
        datasets.append(gratings)

    if half_horizontal.get("include", True):
        target = half_horizontal.get("target", 1)
        uid = half_horizontal.get("uid", None)
        N = half_horizontal.get("N", 2500)
        gratings = GratingsDataset(N=N, vertical=False, half=True, target=target, uid=uid)
        datasets.append(gratings)

    # return concatenated dataset
    return torch.utils.data.ConcatDataset(datasets)
