# Train EmbedSeg model with multiple datasets. Use within directory in which EmbedSeg is installed.
import os
import shutil
import torch
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
import torch.nn.functional as F

import numpy as np

torch.backends.cudnn.benchmark = True

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
    device,
):
    """Trains 3D Model.

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

    device: torch.device

    Returns
    -------
    float
        Average loss
    """
    loss_meter = AverageMeter()
    model.train()

    for param_group in optimizer.param_groups:
        print("learning rate: {}".format(param_group["lr"]))

    for train_dataset_it in train_dataset_it_list:
        for i, sample in enumerate(tqdm(train_dataset_it)):
            im = sample["image"].to(device)  # BCZYX            
            instances = sample["instance"].squeeze(1).to(device)  # BZYX
            class_labels = sample["label"].squeeze(1).to(device)  # BZYX
            center_images = sample["center_image"].squeeze(1).to(device)  # BZYX

                            
            # output = model(im)  # B 7 Z Y X (1 gpu training)
            output = model(**dict(input = im, only_encode=False))  # B 7 Z Y X (multiple gpu training)

            # loss = criterion(output, instances, class_labels, center_images, **args) (1 gpu training)
            loss = criterion(**dict( prediction = output, instances =instances , labels = class_labels, center_images = center_images, w_inst=1, w_var=10, w_seed=1, iou=False, iou_meter=None)) # (multiple gpu training)
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
    
    return loss_meter.avg

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
    device,
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

    device: torch.device

    Returns
    -------
    tuple: (float, float)
        Average loss, Average IoU
    """
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    model.eval()
    with torch.no_grad():
        for val_dataset_it in val_dataset_it_list:
            for i, sample in enumerate(tqdm(val_dataset_it)):
                im = sample["image"].to(device)  # BCZYX
                instances = sample["instance"].squeeze(1).to(device)  # BZYX
                class_labels = sample["label"].squeeze(1).to(device)  # BZYX
                center_images = sample["center_image"].squeeze(1).to(device)  # BZYX
                
                # output = model(im) # (1 gpu training)
                output = model(**dict(input = im, only_encode=False))  # B 7 Z Y X (multiple gpu training)
                
                # loss = criterion(output, instances, class_labels, center_images, **args, iou=True, iou_meter=iou_meter) # (1 gpu training)
                loss = criterion(**dict( prediction = output, instances =instances , labels = class_labels, center_images = center_images, w_inst=1, w_var=10, w_seed=1, iou=True, iou_meter=iou_meter)) # (multiple gpu training)

                loss = loss.mean()
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

    return loss_meter.avg, iou_meter.avg


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
    train_dataset_dict_list,
    val_dataset_dict_list,
    model_dict,
    loss_dict,
    configs,
    color_map="magma",
):
    """Entry function for beginning the model training procedure.

    Parameters
    ----------
    train_dataset_dict_list : [dictionary]
        Dictionary containing training data loader-specific parameters
        (for e.g. train_batch_size etc)
    val_dataset_dict_list : [dictionary]
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
    global train_dataset_it_list, val_dataset_it_list, model, criterion, optimizer, cluster

    # train dataloader
    train_dataset_list = []
    for train_dataset_dict in train_dataset_dict_list:
        train_dataset = get_dataset(
            train_dataset_dict["name"], train_dataset_dict["kwargs"]
        )
        train_dataset_list.append(train_dataset)

    train_dataset_it_list = []
    for i, train_dataset in enumerate(train_dataset_list):
        train_dataset_it = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_dataset_dict_list[i]["batch_size"],
            shuffle=True,
            drop_last=False, # originally true. but needs to be false for dataset size = 1 to be processed
            num_workers=train_dataset_dict_list[i]["workers"],
            pin_memory=True if configs["device"][:4] == "cuda" else False,
        )
        train_dataset_it_list.append(train_dataset_it)

    # val dataloader
    val_dataset_list = []
    for val_dataset_dict in val_dataset_dict_list:
        val_dataset = get_dataset(val_dataset_dict["name"], val_dataset_dict["kwargs"])
        val_dataset_list.append(val_dataset)
    
    val_dataset_it_list = []
    for i, val_dataset in enumerate(val_dataset_list):
        val_dataset_it = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_dataset_dict_list[i]["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=val_dataset_dict_list[i]["workers"],
            pin_memory=True if configs["device"][:4] == "cuda" else False,
        )
        val_dataset_it_list.append(val_dataset_it)

    # set model
    model = get_model(model_dict["name"], model_dict["kwargs"])
    model.init_output(loss_dict["lossOpts"]["n_sigma"])
    # model = model.to(device) (1 gpu training)
    model = torch.nn.DataParallel(model).to(device) # (multiple gpu training)

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
    # criterion = criterion.to(device) # (1 gpu training)
    
    criterion = torch.nn.DataParallel(criterion).to(device) # (multiple gpu training)

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs["train_lr"], weight_decay=1e-4
    )

    def lambda_(epoch):
        return pow((1 - ((epoch) / 400)), 0.9)

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
    logger = Logger(("train", "val", "iou"), "loss")

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
        # logger.data = state["logger_data"]
        logger.data = {k: state["logger_data"][k] for k in ('train', 'val', 'iou')}
    
    for epoch in range(start_epoch, configs["n_epochs"]):
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_, last_epoch=epoch - 1
        )
        print("Starting epoch {}".format(epoch))
        # scheduler.step(epoch)

        train_loss = train_vanilla_3d(
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
            device=device,
        )

        val_loss, val_iou = val_vanilla_3d(
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
            device=device,
        )

        scheduler.step()
        print("===> train loss: {:.2f}".format(train_loss))
        print("===> val loss: {:.2f}, val iou: {:.2f}".format(val_loss, val_iou))

        logger.add("train", train_loss)
        logger.add("val", val_loss)
        logger.add("iou", val_iou)
        logger.plot(save=configs["save"], save_dir=configs["save_dir"])  # TODO

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