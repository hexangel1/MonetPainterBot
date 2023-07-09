"""CycleGAN model"""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tqdm.notebook import tqdm

from utils import (
    unnorm,
    update_req_grad,
    init_weights,
    load_checkpoint,
    save_checkpoint
)


def Upsample(in_ch, out_ch, use_dropout=True, dropout_ratio=0.5):
    """Upsample block"""
    if use_dropout:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.Dropout(dropout_ratio),
            nn.GELU()
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.GELU()
        )


def Convlayer(in_ch, out_ch, kernel_size=3, stride=2, use_leaky=True, use_inst_norm=True, use_pad=True):
    """Convlayer block"""
    if use_pad:
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, 1, bias=True)
    else:
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, 0, bias=True)

    if use_leaky:
        actv = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        actv = nn.GELU()

    if use_inst_norm:
        norm = nn.InstanceNorm2d(out_ch)
    else:
        norm = nn.BatchNorm2d(out_ch)

    return nn.Sequential(conv, norm, actv)


class Resblock(nn.Module):
    """Resblock"""

    def __init__(self, in_features, dropout_ratio=0.5):
        super().__init__()
        layers = list()
        layers.append(nn.ReflectionPad2d(1))
        layers.append(Convlayer(in_features, in_features, 3, 1, False, use_pad=False))
        layers.append(nn.Dropout(dropout_ratio))
        layers.append(nn.ReflectionPad2d(1))
        layers.append(nn.Conv2d(in_features, in_features, 3, 1, padding=0, bias=True))
        layers.append(nn.InstanceNorm2d(in_features))
        self.res = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.res(x)


class Generator(nn.Module):
    """Generator implementation"""

    def __init__(self, in_ch, out_ch, num_res_blocks=10):
        super().__init__()
        layers = list()
        layers.append(nn.ReflectionPad2d(3))
        layers.append(Convlayer(in_ch, 64, 7, 1, False, True, False))
        layers.append(Convlayer(64, 128, 3, 2, False))
        layers.append(Convlayer(128, 256, 3, 2, False))
        for _ in range(num_res_blocks):
            layers.append(Resblock(256))
        layers.append(Upsample(256, 128))
        layers.append(Upsample(128, 64))
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(64, out_ch, kernel_size=7, padding=0))
        layers.append(nn.Tanh())
        self.gen = nn.Sequential(*layers)

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    """Dicriminator implementation"""

    def __init__(self, in_ch, num_layers=6):
        super().__init__()
        layers = list()
        layers.append(nn.Conv2d(in_ch, 64, 4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(1, num_layers):
            in_chs = 64 * 2 ** (i - 1)
            out_chs = in_chs * 2
            if i == num_layers - 1:
                layers.append(Convlayer(in_chs, out_chs, 4, 1))
            else:
                layers.append(Convlayer(in_chs, out_chs, 4, 2))
        layers.append(nn.Conv2d(2048, 1, kernel_size=4, stride=1, padding=1))
        self.disc = nn.Sequential(*layers)

    def forward(self, x):
        return self.disc(x)


class ImageBuffer(object):
    """Image Buffer"""

    def __init__(self, max_imgs=50):
        self.buff_size = max_imgs
        self.buff_used = 0
        self.img_buffer = list()

    def __call__(self, images):
        retval = list()
        for img in images:
            if self.buff_used < self.buff_size:
                self.img_buffer.append(img)
                retval.append(img)
                self.buff_used += 1
            else:
                if np.random.ranf() < 0.5:
                    idx = np.random.randint(0, self.buff_size)
                    retval.append(self.img_buffer[idx])
                    self.img_buffer[idx] = img
                else:
                    retval.append(img)
        return retval


class LRSched():
    """Control learning rate value by during epochs"""

    def __init__(self, decay_epochs=100, total_epochs=200):
        self.decay_epochs = decay_epochs
        self.total_epochs = total_epochs

    def step(self, epoch_num):
        if epoch_num <= self.decay_epochs:
            return 1.0
        else:
            fract = (epoch_num - self.decay_epochs) / (self.total_epochs - self.decay_epochs)
            return 1.0 - fract


class CycleGAN(object):
    """CycleGAN class"""

    def __init__(self, in_ch, out_ch, epochs, device, start_lr=2e-4, alpha=10, beta=5, decay_epoch=-1):
        self.epochs = epochs
        self.decay_epoch = decay_epoch if decay_epoch > 0 else int(epochs / 2)
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.gen_mtp = Generator(in_ch, out_ch)
        self.gen_ptm = Generator(in_ch, out_ch)
        self.disc_m = Discriminator(in_ch)
        self.disc_p = Discriminator(in_ch)
        self.init_models()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.optim_gen = torch.optim.Adam(itertools.chain(self.gen_mtp.parameters(), self.gen_ptm.parameters()),
                                          lr=start_lr, betas=(0.5, 0.999))
        self.optim_disc = torch.optim.Adam(itertools.chain(self.disc_m.parameters(), self.disc_p.parameters()),
                                           lr=start_lr, betas=(0.5, 0.999))
        self.monet_buffer = ImageBuffer()
        self.photo_buffer = ImageBuffer()
        gen_lr = LRSched(self.decay_epoch, self.epochs)
        disc_lr = LRSched(self.decay_epoch, self.epochs)
        self.gen_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.optim_gen, gen_lr.step)
        self.disc_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.optim_disc, disc_lr.step)
        self.gen_losses = []
        self.disc_losses = []

    def init_models(self):
        init_weights(self.gen_mtp)
        init_weights(self.gen_ptm)
        init_weights(self.disc_m)
        init_weights(self.disc_p)
        self.gen_mtp = self.gen_mtp.to(self.device)
        self.gen_ptm = self.gen_ptm.to(self.device)
        self.disc_m = self.disc_m.to(self.device)
        self.disc_p = self.disc_p.to(self.device)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = load_checkpoint(ckpt_path, map_location=self.device)
        self.epochs = ckpt['epoch']
        self.gen_mtp.load_state_dict(ckpt['gen_mtp'])
        self.gen_ptm.load_state_dict(ckpt['gen_ptm'])
        self.disc_m.load_state_dict(ckpt['disc_m'])
        self.disc_p.load_state_dict(ckpt['disc_p'])
        self.optim_gen.load_state_dict(ckpt['optim_gen'])
        self.optim_disc.load_state_dict(ckpt['optim_disc'])

    def save_current_state(self, epoch, ckpt_path):
        save_dict = {
            'epoch': epoch + 1,
            'gen_mtp': self.gen_mtp.state_dict(),
            'gen_ptm': self.gen_ptm.state_dict(),
            'disc_m': self.disc_m.state_dict(),
            'disc_p': self.disc_p.state_dict(),
            'optim_gen': self.optim_gen.state_dict(),
            'optim_disc': self.optim_disc.state_dict()
        }
        save_checkpoint(save_dict, ckpt_path)

    def transfer_style(self, image_path, output_path):
        transform2Tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform2Image = transforms.ToPILImage()

        with Image.open(image_path, "r") as image:
            img_tensor = transform2Tensor(image)
        with torch.no_grad():
            pred_monet = self.gen_ptm(img_tensor.to(self.device)).cpu().detach()
        pred_monet = unnorm(pred_monet)
        image_monet = transform2Image(pred_monet).convert("RGB")
        image_monet.save(output_path, "PNG")

    def train(self, photo_dl):
        for epoch in range(self.epochs):
            avg_gen_loss = 0.0
            avg_disc_loss = 0.0
            t = tqdm(photo_dl, leave=False, total=photo_dl.__len__())
            for _, (real_photo, real_monet) in enumerate(t):
                real_photo, real_monet = real_photo.to(self.device), real_monet.to(self.device)

                # learn generator
                update_req_grad([self.disc_m, self.disc_p], False)
                self.optim_gen.zero_grad()

                # forward pass through generator
                fake_photo = self.gen_mtp(real_monet)
                fake_monet = self.gen_ptm(real_photo)

                cycl_monet = self.gen_ptm(fake_photo)
                cycl_photo = self.gen_mtp(fake_monet)

                id_monet = self.gen_ptm(real_monet)
                id_photo = self.gen_mtp(real_photo)

                monet_disc = self.disc_m(fake_monet)
                photo_disc = self.disc_p(fake_photo)

                # generator losses: identity, adversarial, cycle consistency
                idt_loss_monet = self.l1_loss(id_monet, real_monet) * self.beta
                idt_loss_photo = self.l1_loss(id_photo, real_photo) * self.beta

                cycle_loss_monet = self.l1_loss(cycl_monet, real_monet) * self.alpha
                cycle_loss_photo = self.l1_loss(cycl_photo, real_photo) * self.alpha

                adv_loss_monet = self.mse_loss(monet_disc, torch.ones(monet_disc.size()).to(self.device))
                adv_loss_photo = self.mse_loss(photo_disc, torch.ones(photo_disc.size()).to(self.device))

                # total generator loss
                total_gen_loss = cycle_loss_monet + cycle_loss_photo + \
                    adv_loss_monet + adv_loss_photo + idt_loss_monet + idt_loss_photo

                avg_gen_loss += total_gen_loss.item()

                # backward pass
                total_gen_loss.backward()
                self.optim_gen.step()

                # learn discriminator
                update_req_grad([self.disc_m, self.disc_p], True)
                self.optim_disc.zero_grad()

                # forward pass through discriminator
                fake_monet = self.monet_buffer([fake_monet.cpu().data.numpy()])[0]
                fake_photo = self.photo_buffer([fake_photo.cpu().data.numpy()])[0]
                fake_monet = torch.tensor(fake_monet).to(self.device)
                fake_photo = torch.tensor(fake_photo).to(self.device)

                monet_disc_real = self.disc_m(real_monet)
                monet_disc_fake = self.disc_m(fake_monet)

                photo_disc_real = self.disc_p(real_photo)
                photo_disc_fake = self.disc_p(fake_photo)

                real = torch.ones(monet_disc_real.size()).to(self.device)
                fake = torch.zeros(monet_disc_fake.size()).to(self.device)

                # discriminator losses
                monet_disc_real_loss = self.mse_loss(monet_disc_real, real)
                monet_disc_fake_loss = self.mse_loss(monet_disc_fake, fake)
                photo_disc_real_loss = self.mse_loss(photo_disc_real, real)
                photo_disc_fake_loss = self.mse_loss(photo_disc_fake, fake)

                monet_disc_loss = (monet_disc_real_loss + monet_disc_fake_loss) / 2
                photo_disc_loss = (photo_disc_real_loss + photo_disc_fake_loss) / 2

                # total discriminator loss
                total_disc_loss = monet_disc_loss + photo_disc_loss
                avg_disc_loss += total_disc_loss.item()

                # backward pass
                total_disc_loss.backward()
                self.optim_disc.step()

                t.set_postfix(gen_loss=total_gen_loss.item(), disc_loss=total_disc_loss.item())

            self.save_current_state(epoch, 'current_weights.ckpt')

            self.gen_lr_sched.step()
            self.disc_lr_sched.step()

            avg_gen_loss /= photo_dl.__len__()
            avg_disc_loss /= photo_dl.__len__()
            self.gen_losses.append(avg_gen_loss)
            self.disc_losses.append(avg_disc_loss)

            print(f"Epoch: {epoch+1}/{self.epochs}\n"
                  f"Generator Loss: {avg_gen_loss} | Discriminator Loss: {avg_disc_loss}\n")
