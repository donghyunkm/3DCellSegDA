# Train EmbedSeg model with Adversarial Discriminative Domain Adaptation. Use within directory in which EmbedSeg is installed.
# https://arxiv.org/abs/1702.05464
import os
import shutil

import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from tqdm import tqdm

from EmbedSeg.criterions import get_loss
from EmbedSeg.datasets import get_dataset
from EmbedSeg.models import get_model
from EmbedSeg.utils.utils import (
    AverageMeter,
    Cluster,
    Cluster_3d,
    Logger,
    prepare_embedding_for_train_image,
)

import numpy as np
import time
from EmbedSeg.utils.create_dicts import create_dataset_dict

torch.backends.cudnn.benchmark = True


def domain_adaptation(x_source, x_target, device):
    features_source = model(x_source, only_encode=True).view(x_source.shape[0], -1)
    features_target = model(x_target, only_encode=True).view(x_target.shape[0], -1)
    discriminator_x = torch.cat([features_source, features_target])
    discriminator_y = torch.cat([torch.ones(x_source.shape[0], device=device),
                                 torch.zeros(x_target.shape[0], device=device)])

    logits = discriminator(discriminator_x).squeeze()
    loss = discriminator_criterion(logits, discriminator_y)
    discriminator_optim.zero_grad()
    loss.backward()
    discriminator_optim.step()
    print(logits.shape)
    
    # Train generator (encoder)
    features_target = model(x_target, only_encode=True).view(x_target.shape[0], -1)
    discriminator_y = torch.ones(x_target.shape[0], device=device) # Flipped labels
    logits2 = discriminator(features_target).squeeze()
    loss2 = discriminator_criterion(logits2, discriminator_y)
    optimizer.zero_grad()
    loss2.backward()
    optimizer.step()

    return loss.item()


def train_vanilla_3d(
    display,
    display_embedding,
    display_it,
    one_hot,
    grid_x,
    grid_y,
    grid_z,
    pixel_x,
    pixel_y,
    pixel_z,
    n_sigma,
    zslice,
    args,
    domain_adapt,
    device,
):
    """Trains 3D Model without virtual multiplier.

    Parameters
    ----------
    display : bool
        Displays input, GT, model predictions during training
    display_embedding : bool
        Displays embeddings for train (crop) images
    display_it: int
        Displays a new training image, the corresponding GT
        and model prediction every `display_it` crop images
    one_hot: bool
        In case the GT labels are available in one-hot fashion,
        this parameter is set equal to True
        This parameter is not relevant for 3D data
        and will be deprecated in a future code update
    grid_x: int
        Number of pixels along x dimension which constitute a tile
    grid_y: int
        Number of pixels along y dimension which constitute a tile
    grid_z: int
        Number of pixels along z dimension which constitute a tile
    pixel_x: float
        The grid length along x is mapped to `pixel_x`.
        For example, if grid_x = 1024 and pixel_x = 1.0,
        then each dX or the spacing between consecutive pixels
        along the x dimension is set equal to pixel_x/grid_x = 1.0/1024
    pixel_y: float
        The grid length along y is mapped to `pixel_y`.
        For example, if grid_y = 1024 and pixel_y = 1.0,
        then each dY or the spacing between consecutive pixels
        along the y dimension is set equal to pixel_y/grid_y = 1.0/1024
    pixel_z: float
        The grid length along z is mapped to `pixel_z`.
        For example, if grid_z = 1024 and pixel_z = 1.0,
        then each dY or the spacing between consecutive pixels
        along the z dimension is set equal to pixel_z/grid_z = 1.0/1024
    n_sigma: int
        Should be set equal to 3 for a 3D model
    zslice: int
        If `display` = True,
        then the the raw image at z = z_slice is displayed during training
    args: dictionary
    domain_adapt: bool
    device: torch.device

    Returns
    -------
    float
        Average loss
    """
    loss_meter = AverageMeter()
    loss_meter_domain_adapt = AverageMeter()
    model.train()

    for param_group in optimizer.param_groups:
        print("learning rate: {}".format(param_group["lr"]))
    
    for i, sample in enumerate(tqdm(train_dataset_it)): # Train EmbedSeg model 
        im = sample["image"].to(device)  # BCZYX
        instances = sample["instance"].squeeze(1).to(device)  # BZYX
        class_labels = sample["label"].squeeze(1).to(device)  # BZYX
        center_images = sample["center_image"].squeeze(1).to(device)  # BZYX
        
        output = model(im)  # B 7 Z Y X
        
        loss = criterion(output, instances, class_labels, center_images, **args)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        
        if display and i % display_it == 0:
            with torch.no_grad():
                predictions = cluster.cluster_with_gt(
                    output[0], instances[0], n_sigma=n_sigma
                )
                if one_hot:
                    instance = invert_one_hot(instances[0].cpu().detach().numpy())
                    instance_ids = np.arange(instances.size(1))  # instances[0] --> DYX
                else:
                    instance_ids = instances[0].unique()
                    instance_ids = instance_ids[instance_ids != 0]

    if domain_adapt:
        for i, (sample, sample_target) in enumerate(tqdm(zip(train_dataset_it_2, target_dataset_it))): # Apply domain adaptation with original training data and data in the target domain 
            im = sample["image"].to(device)  # BCZYX
            instances = sample["instance"].squeeze(1).to(device)  # BZYX
            class_labels = sample["label"].squeeze(1).to(device)  # BZYX
            center_images = sample["center_image"].squeeze(1).to(device)  # BZYX
    
            im_target = sample_target['image'].to(device)
            
            loss_domain_adapt = domain_adaptation(im, im_target, device)
            loss_meter_domain_adapt.update(loss_domain_adapt)
            
    
    return loss_meter.avg, loss_meter_domain_adapt.avg


def val_domain_adaptation(x_source, x_target, device):

    with torch.no_grad():
        features_source = model(x_source, only_encode=True).view(x_source.shape[0], -1)
        features_target = model(x_target, only_encode=True).view(x_target.shape[0], -1)

        discriminator_x = torch.cat([features_source, features_target])
        discriminator_y = torch.cat([torch.ones(x_source.shape[0], device=device),
                                     torch.zeros(x_target.shape[0], device=device)])
        logits = discriminator(discriminator_x).squeeze()
        loss = discriminator_criterion(logits, discriminator_y)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct = (preds == discriminator_y).sum().item()
        accuracy = correct / discriminator_y.shape[0]
        
        return loss.item(), accuracy


def val_vanilla_3d(
    display,
    display_embedding,
    display_it,
    one_hot,
    grid_x,
    grid_y,
    grid_z,
    pixel_x,
    pixel_y,
    pixel_z,
    n_sigma,
    zslice,
    args,
    domain_adapt,
    device
):
    """Validates a 3D Model without virtual multiplier.

    Parameters
    ----------
    display : bool
        Displays input, GT, model predictions during training
    display_embedding : bool
        Displays embeddings for train (crop) images
    display_it: int
        Displays a new training image, the corresponding GT
        and model prediction every `display_it` crop images
    one_hot: bool
        In case the GT labels are available in one-hot fashion,
        this parameter is set equal to True
        This parameter is not relevant for 3D data
        and will be deprecated in a future code update
    grid_x: int
        Number of pixels along x dimension which constitute a tile
    grid_y: int
        Number of pixels along y dimension which constitute a tile
    grid_z: int
        Number of pixels along z dimension which constitute a tile
    pixel_x: float
        The grid length along x is mapped to `pixel_x`.
        For example, if grid_x = 1024 and pixel_x = 1.0, then each dX
        or the spacing between consecutive pixels
        along the x dimension is set equal to pixel_x/grid_x = 1.0/1024
    pixel_y: float
        The grid length along y is mapped to `pixel_y`.
        For example, if grid_y = 1024 and pixel_y = 1.0, then each dY
        or the spacing between consecutive pixels
        along the y dimension is set equal to pixel_y/grid_y = 1.0/1024
    pixel_z: float
        The grid length along z is mapped to `pixel_z`.
        For example, if grid_z = 1024 and pixel_z = 1.0, then each dY
        or the spacing between consecutive pixels
        along the z dimension is set equal to pixel_z/grid_z = 1.0/1024
    n_sigma: int
        Should be set equal to 3 for a 3D model
    zslice: int
        If `display` = True, then the the raw image at z = z_slice
        is displayed during training
    args: dictionary
    domain_adapt: bool
    device: torch.device

    Returns
    -------
    tuple: (float, float)
        Average loss, Average IoU
    """
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    gan_loss_meter, gan_acc_meter = AverageMeter(), AverageMeter()
    loss_meter_TL, iou_meter_TL = AverageMeter(), AverageMeter()
    model.eval()
    with torch.no_grad():

        for i, (sample, sample_target, sample_TL) in enumerate(tqdm(zip(val_dataset_it, val_target_dataset_it, val_target_dataset_TL_it))):
            im = sample["image"].to(device)  # BCZYX

            im_target = sample_target["image"].to(device)
            
            instances = sample["instance"].squeeze(1).to(device)  # BZYX
            class_labels = sample["label"].squeeze(1).to(device)  # BZYX
            center_images = sample["center_image"].squeeze(1).to(device)  # BZYX
            output = model(im)

            im_TL = sample_TL['image'].to(device)
            instances_TL = sample_TL["instance"].squeeze(1).to(device)  # BZYX
            class_labels_TL = sample_TL["label"].squeeze(1).to(device)  # BZYX
            center_images_TL = sample_TL["center_image"].squeeze(1).to(device)  # BZYX
            output_TL = model(im_TL)
            
            loss = criterion(
                output,
                instances,
                class_labels,
                center_images,
                **args,
                iou=True,
                iou_meter=iou_meter
            )
            loss = loss.mean()

            # Val TL
            loss_TL = criterion(
                output_TL,
                instances_TL,
                class_labels_TL,
                center_images_TL,
                **args,
                iou=True,
                iou_meter=iou_meter_TL
            )
            loss_TL = loss_TL.mean()

            if domain_adapt:
                gan_loss, gan_acc = val_domain_adaptation(im, im_target, device) 
                gan_loss_meter.update(gan_loss)
                gan_acc_meter.update(gan_acc)
            if display and i % display_it == 0:
                with torch.no_grad():
                    predictions = cluster.cluster_with_gt(
                        output[0], instances[0], n_sigma=n_sigma
                    )
                    if one_hot:
                        instance = invert_one_hot(instances[0].cpu().detach().numpy())
                        instance_ids = np.arange(instances.size(1))
                    else:
                        instance_ids = instances[0].unique()
                        instance_ids = instance_ids[instance_ids != 0]

            loss_meter.update(loss.item())
            loss_meter_TL.update(loss_TL.item())

    return loss_meter.avg, iou_meter.avg, gan_loss_meter.avg, gan_acc_meter.avg, loss_meter_TL.avg, iou_meter_TL.avg


def invert_one_hot(image):
    """Inverts a one-hot label mask.

    Parameters
    ----------
    image : numpy array (I x H x W)
        Label mask present in one-hot fashion
        (i.e. with 0s and 1s and multiple z slices)
        here `I` is the number of GT or predicted objects

    Returns
    -------
    numpy array (H x W)
        A flattened label mask with objects labelled from 1 ... I
    """
    instance = np.zeros((image.shape[1], image.shape[2]), dtype="uint16")
    for z in range(image.shape[0]):
        instance = np.where(image[z] > 0, instance + z + 1, instance)
        # TODO - Alternate ways of inverting one-hot label masks would exist !!
    return instance


def save_checkpoint(
    state, is_best, epoch, save_dir, save_checkpoint_frequency, name="checkpoint.pth"
):
    """Trains 3D Model without virtual multiplier.

    Parameters
    ----------
    state : dictionary
        The state of the model weights
    is_best : bool
        In case the validation IoU is higher at the end of a certain epoch
        than previously recorded, `is_best` is set equal to True
    epoch: int
        The current epoch
    save_checkpoint_frequency: int
        The model weights are saved every `save_checkpoint_frequency` epochs
    name: str, optional
        The model weights are saved under the name `name`

    Returns
    -------

    """
    print("=> saving checkpoint")
    file_name = os.path.join(save_dir, name)
    torch.save(state, file_name)
    if save_checkpoint_frequency is not None:
        if epoch % int(save_checkpoint_frequency) == 0:
            file_name2 = os.path.join(save_dir, str(epoch) + "_" + name)
            torch.save(state, file_name2)
    if is_best:
        shutil.copyfile(file_name, os.path.join(save_dir, "best_iou_model.pth"))


def begin_training(
    train_dataset_dict,
    val_dataset_dict,
    train_dataset_dict_target,
    val_dataset_dict_target,
    val_dataset_dict_target_TL,
    model_dict,
    loss_dict,
    configs,
    domain_adapt,
    color_map="magma",
):
    """Entry function for beginning the model training procedure.

    Parameters
    ----------
    train_dataset_dict : dictionary
        Dictionary containing training data loader-specific parameters
        (for e.g. train_batch_size etc)
    val_dataset_dict : dictionary
        Dictionary containing validation data loader-specific parameters
        (for e.g. val_batch_size etc)
    model_dict: dictionary
        Dictionary containing model specific parameters (for e.g. number of outputs)
    loss_dict: dictionary
        Dictionary containing loss specific parameters
        (for e.g. convex weights of different loss terms - w_iou, w_var etc)
    configs: dictionary
        Dictionary containing general training parameters
        (for e.g. num_epochs, learning_rate etc)
    domain_adapt: bool
    color_map: str, optional
       Name of color map. Used in case configs['display'] is set equal to True

    Returns
    -------
    """

    if configs["save"]:
        if not os.path.exists(configs["save_dir"]):
            os.makedirs(configs["save_dir"])

    if configs["display"]:
        plt.ion()
    else:
        plt.ioff()
        plt.switch_backend("agg")

    # set device
    device = torch.device(configs["device"])

    # define global variables
    global train_dataset_it, val_dataset_it, model, criterion, optimizer, cluster, discriminator, discriminator_optim, discriminator_criterion, target_dataset_it, val_target_dataset_it, train_dataset_it_2, val_target_dataset_TL_it

    # Train dataset contains training data for EmbedSeg (data in source domain)    
    train_dataset = get_dataset(
        train_dataset_dict["name"], train_dataset_dict["kwargs"]
    )

    # Target dataset contains data in the target distribution (used for domain adaptation) 
    target_dataset = get_dataset(
        train_dataset_dict_target["name"], train_dataset_dict_target["kwargs"]
    )

    train_dataset_it = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_dataset_dict["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=train_dataset_dict["workers"],
        pin_memory=True if configs["device"][:4] == "cuda" else False,
    )

    # Note, data in the source domain is used for domain adaptation too. Train_dataset_it_2 is used so that domain adaptation can be done with larger batch size (which is important when training the gan)
    train_dataset_it_2 = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=False,
        num_workers=train_dataset_dict["workers"],
        pin_memory=True if configs["device"][:4] == "cuda" else False,
    )

    target_dataset_it = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=False,
        num_workers=train_dataset_dict["workers"],
        pin_memory=True if configs["device"][:4] == "cuda" else False,
    )

    val_dataset = get_dataset(val_dataset_dict["name"], val_dataset_dict["kwargs"])
    val_dataset_it = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_dataset_dict["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=val_dataset_dict["workers"],
        pin_memory=True if configs["device"][:4] == "cuda" else False,
    )

    val_target_dataset = get_dataset(
        val_dataset_dict_target["name"], val_dataset_dict_target["kwargs"]
    )

    val_target_dataset_it = torch.utils.data.DataLoader(
        val_target_dataset,
        batch_size=val_dataset_dict["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=val_dataset_dict["workers"],
        pin_memory=True if configs["device"][:4] == "cuda" else False,
    )

    # Use target_TL dataset/dataloader to evalaute transfer learning during training
    val_target_dataset_TL = get_dataset(
        val_dataset_dict_target_TL["name"], val_dataset_dict_target_TL["kwargs"]
    )
    val_target_dataset_TL_it = torch.utils.data.DataLoader(
        val_target_dataset_TL,
        batch_size=val_dataset_dict["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=val_dataset_dict["workers"],
        pin_memory=True if configs["device"][:4] == "cuda" else False,
    )

    # set model
    model = get_model(model_dict["name"], model_dict["kwargs"])
    model.init_output(loss_dict["lossOpts"]["n_sigma"])
    model = model.to(device)
    # model = torch.nn.DataParallel(model).to(device)

    # Discriminator for GAN
    discriminator = nn.Sequential(
        nn.Linear(7581, 1280),
        nn.ReLU(),
        nn.Linear(1280, 128),
        nn.ReLU(),
        nn.Linear(128, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr = 0.001)
    discriminator_criterion = nn.BCEWithLogitsLoss()
    
    criterion = get_loss(
        configs["grid_z"],
        configs["grid_y"],
        configs["grid_x"],
        configs["pixel_z"],
        configs["pixel_y"],
        configs["pixel_x"],
        configs["one_hot"],
        loss_dict["lossOpts"],
    )
    criterion = criterion.to(device)
    # criterion = torch.nn.DataParallel(criterion).to(device)

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs["train_lr"], weight_decay=1e-4
    )
    
    def lambda_(epoch):
        # return pow((1 - ((epoch) / 200)), 0.9) # 200 is total # epochs
        return pow((1 - ((epoch) / 400)), 0.9) # 400 max is epochs now

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_, )

    # clustering
    cluster = Cluster_3d(
        configs["grid_z"],
        configs["grid_y"],
        configs["grid_x"],
        configs["pixel_z"],
        configs["pixel_y"],
        configs["pixel_x"],
        device,
        configs["one_hot"],
    )

    # Logger
    logger = Logger(("train", "val", "iou", 'val_gan_acc', 'TL_iou' ), "loss")

    # resume
    start_epoch = 0
    best_iou = 0
    if configs["resume_path"] is not None and os.path.exists(configs["resume_path"]):
        print("Resuming model from {}".format(configs["resume_path"]))
        state = torch.load(configs["resume_path"])
        start_epoch = 0 
        best_iou = state["best_iou"]
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optim_state_dict"])
        logger.data = state["logger_data"]

    for epoch in range(start_epoch, configs["n_epochs"]):

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_, last_epoch=epoch - 1
        )
        print("Starting epoch {}".format(epoch))
        # scheduler.step(epoch)
        start_time = time.time()  
        train_loss, train_gan_loss = train_vanilla_3d(
            display=configs["display"],
            display_embedding=configs["display_embedding"],
            display_it=configs["display_it"],
            one_hot=configs["one_hot"],
            n_sigma=loss_dict["lossOpts"]["n_sigma"],
            zslice=configs["display_zslice"],
            grid_x=configs["grid_x"],
            grid_y=configs["grid_y"],
            grid_z=configs["grid_z"],
            pixel_x=configs["pixel_x"],
            pixel_y=configs["pixel_y"],
            pixel_z=configs["pixel_z"],
            args=loss_dict["lossW"],
            domain_adapt=domain_adapt,
            device=device,
        )

        val_loss, val_iou, val_gan_loss, val_gan_acc, TL_loss, TL_iou = val_vanilla_3d(
            display=configs["display"],
            display_embedding=configs["display_embedding"],
            display_it=configs["display_it"],
            one_hot=configs["one_hot"],
            n_sigma=loss_dict["lossOpts"]["n_sigma"],
            zslice=configs["display_zslice"],
            grid_x=configs["grid_x"],
            grid_y=configs["grid_y"],
            grid_z=configs["grid_z"],
            pixel_x=configs["pixel_x"],
            pixel_y=configs["pixel_y"],
            pixel_z=configs["pixel_z"],
            args=loss_dict["lossW"],
            domain_adapt=domain_adapt,
            device=device,
        )

        scheduler.step()
        print("===> train loss: {:.2f}".format(train_loss))
        print("===> val loss: {:.2f}, val iou: {:.2f}".format(val_loss, val_iou))
        print("===> val gan acc: {:.2f}, val gan loss: {:.2f}".format(val_gan_acc, val_gan_loss))
        print("===> TL loss: {:.2f}, TL iou: {:.2f}".format(TL_loss, TL_iou))

        logger.add("train", train_loss)
        # logger.add("train_gan", train_gan_loss)

        logger.add("val", val_loss)
        logger.add("iou", val_iou)
        # logger.add("val_gan_loss", val_gan_loss)
        logger.add("val_gan_acc", val_gan_acc)
        logger.add("TL_iou", TL_iou)

        logger.plot(save=configs["save"], save_dir=configs["save_dir"])  # TODO

        end_time = time.time()  
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch+1} took {epoch_duration:.2f} seconds")
        
        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)

        if configs["save"]:
            state = {
                "epoch": epoch,
                "best_iou": best_iou,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "logger_data": logger.data,
            }
        save_checkpoint(
            state,
            is_best,
            epoch,
            save_dir=configs["save_dir"],
            save_checkpoint_frequency=configs["save_checkpoint_frequency"],
        )
