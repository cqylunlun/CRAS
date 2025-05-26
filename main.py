from datetime import datetime

import os
import logging
import sys
import click
import torch
import warnings
import backbones
import cras
import utils


@click.group(chain=True)
@click.option("--results_path", type=str, default="results")
@click.option("--gpu", type=int, default=[], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--test", type=str, default="ckpt")
def main(**kwargs):
    pass


@main.command("net")
@click.option("--train_backbone", is_flag=True)
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1536)
@click.option("--target_embed_dimension", type=int, default=1536)
@click.option("--patchsize", type=int, default=3)
@click.option("--meta_epochs", type=int, default=100)
@click.option("--eval_epochs", type=int, default=1)
@click.option("--dsc_layers", type=int, default=3)
@click.option("--pre_proj", type=int, default=1)
@click.option("--noise", type=float, default=0.015)
@click.option("--k", type=float, default=0.3)
@click.option("--lr", type=float, default=0.0001)
@click.option("--limit", type=int, default=392)
def net(
        backbone_names,
        layers_to_extract_from,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize,
        meta_epochs,
        eval_epochs,
        dsc_layers,
        train_backbone,
        pre_proj,
        noise,
        k,
        lr,
        limit,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = []
        for idx in range(len(backbone_names)):
            layers_to_extract_from_coll.append(layers_to_extract_from)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_cras(input_shape, device):
        crases = []
        for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            cras_inst = cras.CRAS(device)
            cras_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                meta_epochs=meta_epochs,
                eval_epochs=eval_epochs,
                dsc_layers=dsc_layers,
                train_backbone=train_backbone,
                pre_proj=pre_proj,
                noise=noise,
                k=k,
                lr=lr,
                limit=limit,
            )
            crases.append(cras_inst.to(device))
        return crases

    return "get_cras", get_cras


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=str)
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--setting", default="multi", show_default=True)
@click.option("--batch_size", default=32, type=int, show_default=True)
@click.option("--resize", default=329, type=int, show_default=True)
@click.option("--imagesize", default=288, type=int, show_default=True)
@click.option("--num_workers", default=16, type=int, show_default=True)
def dataset(
        name,
        data_path,
        subdatasets,
        setting,
        batch_size,
        resize,
        imagesize,
        num_workers,
):
    _DATASETS = {"mvtec": ["datasets.mvtec", "MVTecDataset"],
                 "visa": ["datasets.mvtec", "MVTecDataset"],
                 "mpdd": ["datasets.mvtec", "MVTecDataset"],
                 "itdd": ["datasets.mvtec", "MVTecDataset"], }
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed, get_name=name):
        if setting == "multi":
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdatasets,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            train_dataloader.name = get_name + "_dataset"

        train_list = []
        train_len_list = []
        for subdataset in subdatasets:  # data for center init
            train_dataset_ = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=[subdataset],
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
            )

            train_dataloader_ = torch.utils.data.DataLoader(
                train_dataset_,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            train_dataloader_.name = get_name + "_" + subdataset
            train_list.append(train_dataloader_)
            train_len_list.append(len(train_dataset_))

        test_list = []
        test_len_list = []
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=[subdataset],
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            test_dataloader.name = get_name + "_" + subdataset
            test_list.append(test_dataloader)
            test_len_list.append(len(test_dataset))

        dataloaders = []
        if setting == "multi":  # multi-class IAD
            dataloader_dict = {
                "training": train_dataloader,
                "trainings": train_list,
                "testing": test_list,
                "classnames": subdatasets,
                "setting": setting,
            }
            dataloaders.append(dataloader_dict)
            LOGGER.info(f"Dataset {'ALL':^20}: train={len(train_dataset)} test={sum(test_len_list)}")
            print("\n")
            for subdataset, train_len, test_len in zip(subdatasets, train_len_list, test_len_list):
                LOGGER.info(f"Dataset {subdataset.upper():^20}: train={train_len} test={test_len}")

        else:  # single-class IAD
            for train_dataloader, train_len, test_dataloader, test_len, subdataset \
                    in zip(train_list, train_len_list, test_list, test_len_list, subdatasets):
                dataloader_dict = {
                    "training": train_dataloader,
                    "trainings": [train_dataloader],
                    "testing": [test_dataloader],
                    "classnames": [subdataset],
                    "setting": setting,
                }
                dataloaders.append(dataloader_dict)
                LOGGER.info(f"Dataset {subdataset:^20}: train={train_len} test={test_len}")

        return dataloaders

    return "get_dataloaders", get_dataloaders


@main.result_callback()
def run(
        methods,
        results_path,
        gpu,
        seed,
        test,
):
    methods = {key: item for (key, item) in methods}
    run_save_path = utils.create_storage_folder(results_path)
    list_of_dataloaders = methods["get_dataloaders"](seed)
    device = utils.set_torch_device(gpu)

    result_collect = []
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        utils.fix_seeds(seed, device)

        imagesize = dataloaders["training"].dataset.imagesize
        cras_list = methods["get_cras"](imagesize, device)

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        for i, CRAS in enumerate(cras_list):
            flag = 0., 0., 0., 0., 0., -1.
            if CRAS.backbone.seed is not None:
                utils.fix_seeds(CRAS.backbone.seed, device)

            CRAS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataloaders["training"].name)
            if test == 'ckpt':
                print("\n")
                LOGGER.info(
                    "Train-Selecting dataset [{}] ({}/{}) {}".format(
                        dataloaders["training"].name,
                        dataloader_count + 1,
                        len(list_of_dataloaders),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    )
                )
                flag = CRAS.trainer(dataloaders["training"], dataloaders["trainings"],
                                    dataloaders["testing"], dataloaders["classnames"], dataloaders["setting"])

            if type(flag) != int:
                for num, test_loader in enumerate(dataloaders["testing"]):
                    print("\n")
                    LOGGER.info(
                        "Test-Selecting dataset [{}] ({}/{}) {}".format(
                            test_loader.name,
                            num + 1,
                            len(dataloaders["testing"]),
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        )
                    )
                    i_auroc, i_ap, p_auroc, p_ap, p_pro, epoch = CRAS.tester(test_loader, dataloaders["classnames"])
                    result_collect.append(
                        {
                            "dataset_name": test_loader.name,
                            "image_auroc": i_auroc,
                            "image_ap": i_ap,
                            "pixel_auroc": p_auroc,
                            "pixel_ap": p_ap,
                            "pixel_pro": p_pro,
                            "best_epoch": epoch,
                        }
                    )

                    if epoch > -1:
                        for key, item in result_collect[-1].items():
                            if isinstance(item, str):
                                continue
                            elif isinstance(item, int):
                                print(f"{key}:{item}")
                            else:
                                print(f"{key}:{round(item * 100, 2)} ", end="")

                    mean_metrics = utils.create_csv(result_collect, run_save_path)

                    if num == len(dataloaders["classnames"]) - 1 and len(dataloaders["classnames"]) != 1:
                        print("\n")
                        LOGGER.info("Mean metrics {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                        for key, item in mean_metrics.items():
                            if 'epoch' in key:
                                print(f"{key[5:]}:{round(item)}")
                            else:
                                print(f"{key[5:]}:{round(item * 100, 2)} ", end="")
                        print("\n")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
