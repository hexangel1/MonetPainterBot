""" Train """
import torch
from torch.utils.data import DataLoader

from model import CycleGAN
from dataset import ImageDataset
from utils import set_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(719)

img_ds = ImageDataset('../input/gan-getting-started/monet_jpg/', '../input/gan-getting-started/photo_jpg/')
img_dl = DataLoader(img_ds, batch_size=1, pin_memory=True)
gan = CycleGAN(3, 3, 50, device)
gan.train(img_dl)