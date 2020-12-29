import sys

sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from GAN.han_gan.gan import *
from gan_datamodules import *
import joblib
import pickle
import argparse
from lang import *
from document.han.han import *
from snli.bilstm.bilstm import *
from snli.attn_enc.attn_enc import *
from joblib import Memory
import shutil
import pytorch_lightning as pl
import json

from utils.load_models import (
    load_bilstm_encoder,
    load_attn_encoder,
    load_han_clf_encoder,
    load_han_reg_encoder,
)
from utils.helpers import seed_torch
import os
from utils.keys import NEPTUNE_API
from tqdm.auto import tqdm
import neptune


def train_model(args):
    # initiate neptune logger

    if args.log:
        neptune.init(
            project_qualified_name="aparkhi/NoveltyGAN",
            api_token=NEPTUNE_API,
        )
        neptune.create_experiment()

    EPOCHS = args.epochs
    batch_size = 16
    seed_torch()

    ### get models
    G, D, Encoder, lang, num_sent = get_models(args)
    G[0] = G[0].cuda()
    D[0] = D[0].cuda()
    Encoder[0] = Encoder[0].cuda()
    print("GAN Initialized")

    # data module
    if args.webis:
        data_module = WebisDataModule(batch_size=batch_size)
        data_module.prepare_data(lang, num_sent, labeled_samples=args.labeled_size)
        print("Data Module Initialzied")
    if args.dlnd:
        data_module = DLNDDataModule(batch_size=batch_size)
        data_module.prepare_data(lang, num_sent, labeled_samples=args.labeled_size)
        print("Data Module Initialzied")

    ### get dataloaders
    all_train, labeled_train, val, test = get_dataloader(data_module)
    labeled_train = create_infinite_dataloader(labeled_train)

    ### Params
    params = {
        "batch_size": batch_size,
        "generator_lr": 3e-4,
        "discriminator_lr": 3e-4,
    }
    if args.log:
        neptune.log_text("params", json.dumps(params))
        neptune.log_text("Generator_conf", json.dumps(vars(G[1])))
        neptune.log_text("Discriminator_conf", json.dumps(vars(D[1])))
        # neptune.log_text('Encoder_conf',json.dumps(vars(Encoder[1])))

    # Configure optimizers
    opt_gen = torch.optim.Adam(G[0].parameters(), lr=params["generator_lr"])
    opt_disc = torch.optim.Adam(D[0].parameters(), lr=params["discriminator_lr"])

    # Each epoch
    print("Training Started")
    for i in range(EPOCHS):
        gen_loss, disc_loss = train_epoch(
            G,
            D,
            Encoder,
            (all_train, labeled_train),
            (opt_gen, opt_disc),
            params,
            args.log,
        )
        val_loss, val_acc = val_epoch(G, D, Encoder, val, args.log)

    test_loss, test_acc = test_epoch(G, D, Encoder, test, args.log)
    return test_acc


def val_epoch(Generator, Discriminator, Encoder, dataloader, log):
    with torch.no_grad():
        G, G_conf = Generator
        D, D_conf = Discriminator
        E, E_conf = Encoder

        G.eval()
        D.eval()
        E.eval()

        pbar = tqdm(dataloader)
        num_correct = total_samples = 0
        XE = nn.CrossEntropyLoss().cuda()
        avg_loss = 0

        for x0, x1, y in pbar:

            x0 = x0.cuda()
            x1 = x1.cuda()
            y = y.cuda()

            encoded_inp = E(x0, x1)
            layer_lab, logits_lab, prob_lab = D(encoded_inp)
            encoded_inp = Variable(encoded_inp)
            pred_lab = torch.argmax(prob_lab, dim=1)
            num_correct += torch.sum(pred_lab == y)
            total_samples += len(y)

            loss = XE(prob_lab, y)
            avg_loss += loss.item()
            pbar.set_postfix(
                {"Validation Loss": loss.item()}
            )  # , "GPU": get_gpu_mem()})

        avg_loss /= len(dataloader)
        acc = num_correct.item() / total_samples
        if log:
            neptune.log_metric("Validation Loss", avg_loss)
            neptune.log_metric("Validation Accuracy", acc)
        print("Val Accuracy:", acc)
        return avg_loss, acc


def test_epoch(Generator, Discriminator, Encoder, dataloader, log):
    with torch.no_grad():
        G, G_conf = Generator
        D, D_conf = Discriminator
        E, E_conf = Encoder

        G.eval()
        D.eval()
        E.eval()

        pbar = tqdm(dataloader)
        num_correct = total_samples = 0
        XE = nn.CrossEntropyLoss().cuda()
        avg_loss = 0

        for x0, x1, y in pbar:

            x0 = x0.cuda()
            x1 = x1.cuda()
            y = y.cuda()

            encoded_inp = E(x0, x1)
            layer_lab, logits_lab, prob_lab = D(encoded_inp)
            encoded_inp = Variable(encoded_inp)
            pred_lab = torch.argmax(prob_lab, dim=1)
            num_correct += torch.sum(pred_lab == y)
            total_samples += len(y)

            loss = XE(prob_lab, y)
            avg_loss += loss.item()
            pbar.set_postfix({"Test Loss": loss.item()})  # , "GPU": get_gpu_mem()})

        avg_loss /= len(dataloader)
        acc = num_correct.item() / total_samples
        if log:
            neptune.log_metric("Test Loss", avg_loss)
            neptune.log_metric("Test Accuracy", acc)
        print("Test Accuracy:", acc)
        return avg_loss, acc


def train_epoch(
    Generator, Discriminator, Encoder, dataloaders, optimizers, params, log
):

    G, G_conf = Generator
    D, D_conf = Discriminator
    E, E_conf = Encoder

    all_train, labeled_train = dataloaders

    opt_gen, opt_disc = optimizers
    XE = nn.CrossEntropyLoss().cuda()

    G.train()
    D.train()
    E.train()
    avg_gen_loss = avg_disc_loss = 0

    pbar = tqdm(all_train)
    for unlab_train_x0, unlab_train_x1, _ in pbar:
        lab_train_x0, lab_train_x1, lab_train_y = next(labeled_train)

        unl_x0 = unlab_train_x0.cuda()
        unl_x1 = unlab_train_x1.cuda()

        inp_x0 = lab_train_x0.cuda()
        inp_x1 = lab_train_x1.cuda()

        lbl = lab_train_y.cuda()

        z = torch.randn(params["batch_size"], G_conf.latent_dim).cuda()

        # Train Discriminator
        opt_disc.zero_grad()
        gen_inp = G(z)
        encoded_inp = E(inp_x0, inp_x1)
        encoded_inp = Variable(encoded_inp)
        layer_lab, logits_lab, prob_lab = D(encoded_inp)
        layer_fake, logits_gen, prob_gen = D(gen_inp)

        encoded_unl = E(unl_x0, unl_x1)
        layer_real, logits_unl, prob_unl = D(encoded_unl)
        l_unl = torch.logsumexp(logits_unl, dim=1)
        l_gen = torch.logsumexp(logits_gen, dim=1)
        loss_unl = (
            0.5 * torch.mean(F.softplus(l_unl))
            - 0.5 * torch.mean(l_unl)
            + 0.5 * torch.mean(F.softplus(l_gen))
        )
        loss_lab = torch.mean(XE(logits_lab, lbl))
        loss_disc = 0.5 * loss_lab + 0.5 * loss_unl
        loss_disc.backward()
        opt_disc.step()
        avg_disc_loss += loss_disc

        # Train Generator
        opt_gen.zero_grad()
        opt_disc.zero_grad()
        gen_inp = G(z)
        layer_fake, _, _ = D(gen_inp)
        encoded_unl = E(unl_x0, unl_x1)
        layer_real, _, _ = D(encoded_unl)
        m1 = torch.mean(layer_real, dim=0)
        m2 = torch.mean(layer_fake, dim=0)
        loss_gen = torch.mean((m1 - m2) ** 2)
        loss_gen.backward()
        opt_gen.step()
        avg_gen_loss += loss_gen
        pbar.set_postfix(
            {
                "Generator loss": loss_gen.item(),
                "Discriminator loss": loss_disc.item(),
                # "GPU": get_gpu_mem(),
            }
        )
        if log:
            neptune.log_metric("Generator Loss", loss_gen.item())
            neptune.log_metric("Discriminator Loss", loss_disc.item())

    avg_gen_loss /= len(all_train)
    avg_disc_loss /= len(all_train)
    return avg_gen_loss, avg_disc_loss


def get_gpu_mem():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp/gpu_free")
    memory_available = [
        int(x.split()[2]) for x in open("./tmp/gpu_free", "r").readlines()
    ]
    return memory_available[0]


def create_infinite_dataloader(dataloader):
    data_iter = iter(dataloader)
    while True:
        try:
            yield next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)


def get_models(args):
    # Generator
    G_conf = Generator_conf()
    G = Generator(G_conf)

    # Discriminator
    D_conf = Discriminator_conf()
    D = Discriminator(D_conf)

    # doc encoder (trained on IMDB)
    doc_enc, lang = load_han_clf_encoder()
    doc_enc_conf = doc_enc.conf
    num_sent = doc_enc_conf.num_sent

    # Doc Pair Encoder (For Novelty Detection)
    Encoder_conf = HAN_GAN_encoder_conf(encoder=doc_enc)
    Encoder = HAN_GAN_encoder(Encoder_conf)

    return [G, G_conf], [D, D_conf], [Encoder, Encoder_conf], lang, num_sent


def get_dataloader(datamodule):
    """Returns dataloaders for unlabeled_train, labeled_train, validation, testing

    Args:
        datamodule ([pl.LightningDataModule]): Datamodule

    Returns:
        all_train [type]: unlabeled_train,
        labeled_train [type]: labeled_train,
        val [type]: validation,
        test [type]: testing,
    """
    all_train = datamodule.train_dataloader()
    labeled_train = datamodule.labeled_dataloader()
    val = datamodule.val_dataloader()
    test = datamodule.test_dataloader()
    return all_train, labeled_train, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN-HAN Training")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    parser.add_argument("--epochs", type=int, default=1, help="Encoder Type")
    parser.add_argument("--webis", action="store_true", help="Webis dataset")
    parser.add_argument("--dlnd", action="store_true", help="DLND dataset")
    parser.add_argument("--labeled_size", type=int, default=5, help="Encoder Type")
    parser.add_argument("--train_size", type=float, default=0.8, help="Encoder Type")
    parser.add_argument("--train_size", type=float, default=0.8, help="Encoder Type")
    parser.add_argument(
        "--log", action="store_true", help="Webis dataset", default=True
    )
    args = parser.parse_args()
    test_acc = train_model(args)
