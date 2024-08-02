import os, argparse, random, time, csv;

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from .ddlk.ddlk import ddlk, mdn, swap
from .ddlk import utils

def convert_X_to_loader(X, get_torch_loaders=True, batch_size=128, num_workers=4):
    full_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32));
    if get_torch_loaders:
        full_loader = DataLoader(full_dataset, batch_size, shuffle=True, num_workers=num_workers);
        return full_loader;
    return full_dataset;

def main(args):
    logger.info('input_url={}'.format(args.input_url));
    assert os.path.exists(args.input_url);

    X = np.genfromtxt(args.input_url, delimiter=',', skip_header=0, encoding='ascii');
    X = np.asarray(X, dtype=float);
    n, p = X.shape;
    logger.info('X={}'.format(X.shape, ));

    knockoff_url = args.input_url.replace('.csv', '_knockoff_DDLK.csv');
    if os.path.isfile(knockoff_url): return;

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpus = 1 if torch.cuda.is_available() else 0

    cp_dir = os.path.join(os.path.dirname(args.input_url), "DDLK_checkpoints")
    if not os.path.exists(cp_dir): os.makedirs(cp_dir)

    def get_mdn_cp_dir(epochs):
        return os.path.join(cp_dir, "MDNJoint_epoch{}_batch{}".format(epochs, args.batch_size,));

    def get_ddlk_cp_dir(epochs):
        return os.path.join(cp_dir, "DDLK_MDNEpochs{}_DDLKEpochs{}_batch{}".format(args.MDN_epochs, epochs, args.batch_size,));

    # Joint distribution model cp url
    mdn_dir = get_mdn_cp_dir(epochs=args.MDN_epochs)
    # knockoff generator model cp url
    ddlk_dir = get_ddlk_cp_dir(epochs=args.DDLK_epochs)

    # Get Train hparams
    dataloader = convert_X_to_loader(X, batch_size=args.batch_size);

    ((X_mu, ), (X_sigma, )) = utils.get_two_moments(dataloader);
    hparams = argparse.Namespace(X_mu=X_mu, X_sigma=X_sigma);

    # train joint distribution:
    if os.path.exists(mdn_dir):
        mdn_url = os.path.join(mdn_dir, "last.ckpt")
        q_joint = mdn.MDNJoint.load_from_checkpoint(mdn_url, hparams=hparams)
        logger.info(f"load existing MDNJoint model from={mdn_url}")
    else:
        q_joint = mdn.MDNJoint(hparams)
        prev_mdn_url=None
        for e in range(args.MDN_epochs, 0, -1):
            prev_mdn_dir = get_mdn_cp_dir(e)
            if os.path.isdir(prev_mdn_dir):
                os.rename(prev_mdn_dir, mdn_dir)
                prev_mdn_url = os.path.join(mdn_dir, "last.ckpt")
                logger.info(f"Resume MDNJoint trainning from epoch={e} checkpoint url={prev_mdn_url}")

        logger.info(f"========== start training Joint Distribution ==========")

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=mdn_dir,
            filename='MDNJoint_{epoch:02d}_{val_loss:.2f}',
            save_last=True,
            save_top_k=1,
            mode='min',
        )
        trainer = pl.Trainer(
            resume_from_checkpoint=prev_mdn_url,
            max_epochs=args.MDN_epochs,
            num_sanity_val_steps=1,
            weights_summary=None,
            callbacks=[checkpoint_callback],
            gpus=gpus, logger=False
        )
        trainer.fit(q_joint, train_dataloader=dataloader, val_dataloaders=[dataloader])

    # train knockoff generator
    hparams = argparse.Namespace(X_mu=X_mu, X_sigma=X_sigma)
    if os.path.exists(ddlk_dir):
        ddlk_url = os.path.join(ddlk_dir, "last.ckpt")
        q_knockoff = ddlk.DDLK.load_from_checkpoint(
            ddlk_url, hparams=hparams, q_joint=q_joint
        )
        logger.info(f"load existing DDLK model from={ddlk_url}")
    else:
        q_knockoff = ddlk.DDLK(hparams, q_joint=q_joint)
        prev_ddlk_url=None
        for e in range(args.DDLK_epochs, 0, -1):
            prev_ddlk_dir = get_ddlk_cp_dir(e)
            if os.path.isdir(prev_ddlk_dir):
                os.rename(prev_ddlk_dir, ddlk_dir)
                prev_ddlk_url = os.path.join(ddlk_dir, "last.ckpt")
                logger.info(f"Resume DDLK trainning from epoch={e} checkpoint url={prev_ddlk_url}")
                break
        logger.info(f"========== start training Knockoff Generator ==========")

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=ddlk_dir,
            filename='DDLK_{epoch:02d}_{val_loss:.2f}',
            save_last=True,
            save_top_k=1,
            mode='min',
        )
        trainer = pl.Trainer(
            resume_from_checkpoint=prev_ddlk_url,
            callbacks=[checkpoint_callback],
            max_epochs=args.DDLK_epochs,
            checkpoint_callback=True,
            logger=False, gpus=gpus,
        )
        trainer.fit(q_knockoff, train_dataloader=dataloader, val_dataloaders=[dataloader])

    X_knockoff = q_knockoff.sample(torch.tensor(X)).detach().numpy()
    assert X_knockoff.shape == X.shape;
    logger.info('X_knockoff={}'.format(X_knockoff.shape, ));
    np.savetxt(knockoff_url, X_knockoff, delimiter=",", fmt='%.6f');

def DDLK(X, batch_size=128, MDN_epochs=50, DDLK_epochs=50):
    dataloader = convert_X_to_loader(X, batch_size=batch_size)
    ((X_mu, ), (X_sigma, )) = utils.get_two_moments(dataloader)
    hparams = argparse.Namespace(X_mu=X_mu, X_sigma=X_sigma)
    q_joint = mdn.MDNJoint(hparams)
    trainer = pl.Trainer(max_epochs=MDN_epochs, logger=False)
    trainer.fit(q_joint, train_dataloaders=dataloader)


    hparams = argparse.Namespace(X_mu=X_mu, X_sigma=X_sigma)
    q_knockoff = ddlk.DDLK(hparams, q_joint=q_joint)
    trainer = pl.Trainer(max_epochs=DDLK_epochs, logger=False)
    trainer.fit(q_knockoff, train_dataloaders=dataloader)
    X_knockoff = q_knockoff.sample(torch.tensor(X)).detach().numpy()
    return X_knockoff


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Optional app description')
#     # data params
#     parser.add_argument('--input_url', type=str, help='input_url', required=True);
#
#     parser.add_argument('--batch_size', type=int, help="batch size", default=128)
#     parser.add_argument('--MDN_epochs', type=int, help="number of epochs for training joint distribution", default=50)
#     parser.add_argument('--DDLK_epochs', type=int, help="number of epochs for training knockoff generator", default=50)
#
#     args = parser.parse_args()
#     main(args)
#
