import os
from PIL import Image
from torch.utils.data import Dataset
from taming.data.base import NumpyPaths

class MVTecBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex


class MVTecTrain(MVTecBase):
    def __init__(self, root , size, keys=None):
        super().__init__()
        # Assurez-vous que la liste des chemins est correctement récupérée pour MVTec
        paths = [os.path.join(root, fname) for fname in os.listdir(root)]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=True)
        self.keys = keys


class MVTecValidation(MVTecBase):
    def __init__(self, root, size, keys=None):
        super().__init__()
       
        # Assurez-vous que la liste des chemins est correctement récupérée pour MVTec
        paths = [os.path.join(root, fname) for fname in os.listdir(root)]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys
