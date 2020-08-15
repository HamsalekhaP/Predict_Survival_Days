# Code structure and organistion for this file is inspired by https://www.manning.com/books/deep-learning-with-pytorch, Part 2

# This code is responsible for building the DNN, training it and saving the train and validation results to visualize
# on Tensorboard

# standard imports
import time
import os
import datetime

# other library imports
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# local imports
import config as CONFIG
from model.metaNet import MetaNet
from src.data.data_process import MetsDataset


class RegTraining:
    def __init__(self):
        print(f'In train class with arguments {CONFIG.cli_args}')

        self.model = self.initialiseModel()
        self.optimizer = self.initialiseOptimizer()
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # instantiate Summarywriter for train and validation plots
        self.trn_writer = None
        self.val_writer = None
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        # borrowed from https://www.manning.com/books/deep-learning-with-pytorch, Part 2, Chapter 12
        self.augmentation_dict = {}

        if CONFIG.cli_args.augmented or CONFIG.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if CONFIG.cli_args.augmented or CONFIG.cli_args.augment_offset:
            self.augmentation_dict['offset'] = CONFIG.cli_args.augment_offset
        if CONFIG.cli_args.augmented or CONFIG.cli_args.augment_scale:
            self.augmentation_dict['scale'] = CONFIG.cli_args.augment_scale
        if CONFIG.cli_args.augmented or CONFIG.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = CONFIG.cli_args.augment_rotate
        if CONFIG.cli_args.augmented or CONFIG.cli_args.augment_noise:
            self.augmentation_dict['noise'] = CONFIG.cli_args.augment_noise
        # borrowed from https://www.manning.com/books/deep-learning-with-pytorch, Part 2

    def initialiseModel(self):
        model = MetaNet()

        total_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in model.parameters())
        print(f'{total_params} trainable parameters out of a total of {total_trainable_params} parameters ')

        if CONFIG.USE_CUDA:
            print("Using {} CUDA devices.".format(CONFIG.CUDA_DEVICE_COUNT))
            if CONFIG.CUDA_DEVICE_COUNT > 1:
                # for parallel processing if more GPU available
                model = nn.DataParallel(model)
            model = model.to(CONFIG.DEVICE)
        return model

    def initialiseOptimizer(self):
        # return SGD(self.model.parameters(), lr=CONFIG.Lr, momentum=0.99)
        return Adam(self.model.parameters(), weight_decay=1e-05, lr=CONFIG.Lr)

    def initialiseTrainDataloader(self):
        train_ds = MetsDataset(
            val_stride=10,
            isValSet_bool=False,
            augmentation_dict=self.augmentation_dict,
        )

        batch_size = CONFIG.cli_args.batch_size

        # account for batchsize with multiple processors
        if CONFIG.USE_CUDA:
            batch_size *= CONFIG.CUDA_DEVICE_COUNT

        # pin_memory speed up host device while transferring data from  CPU to GPU
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=CONFIG.cli_args.num_workers,
            pin_memory=CONFIG.USE_CUDA,
            # drop_last= True,
            # shuffle=True
        )

        return train_loader

    def initialiseValDataloader(self):
        val_ds = MetsDataset(
            val_stride=10,
            isValSet_bool=True,
        )
        batch_size = CONFIG.cli_args.batch_size
        if CONFIG.USE_CUDA:
            batch_size *= CONFIG.CUDA_DEVICE_COUNT

        # pin_memory speed up host device while transferring data from  CPU to GPU
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=CONFIG.cli_args.num_workers,
            pin_memory=CONFIG.USE_CUDA,
            # drop_last= True,
            # shuffle=True
        )
        return val_loader

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()

        start_time = time.time()

        # for each batch, compute gradient and update
        for batch_ndx, batch_tup in enumerate(train_dl):
            self.optimizer.zero_grad()

            train_loss_var = self.computeBatchLoss(
                batch_tup
            )

            train_loss_var.backward()
            self.optimizer.step()

            print(f'TRAINING EPOCH: {epoch_ndx:03d}/{CONFIG.cli_args.epochs:03d} | '
                  f'Batch {batch_ndx + 1:03d}/{len(train_dl):03d} |'
                  f'------------------ Cost: {train_loss_var:.4f} | '
                  f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')
        print(f'Total Training Time: {(time.time() - start_time) / 60 :.2f} min')

        return train_loss_var.detach()

    def doValidation(self, epoch_ndx, val_dl):
        # turn off autograd and evaluate model
        with torch.no_grad():
            self.model.eval()
            # for each batch, compute loss
            for batch_ndx, batch_tup in enumerate(val_dl):
                val_loss_var = self.computeBatchLoss(batch_tup)

                print(f'VALIDATION EPOCH: {epoch_ndx:03d}/{CONFIG.cli_args.epochs:03d} | '
                      f'Batch {batch_ndx + 1:03d}/{len(val_dl):03d} |'
                      f'------------------ Cost: {val_loss_var:.4f} | ')

        return val_loss_var.detach()

    def computeBatchLoss(self, batch_tup):
        input_t, label_t, patient_id, age_t, slice_list = batch_tup

        input_d = input_t.to(CONFIG.DEVICE, non_blocking=True)
        age_d = age_t.to(CONFIG.DEVICE, non_blocking=True)
        label_d = label_t.to(CONFIG.DEVICE, non_blocking=True)

        # pass input to instantiated model and retrieve regression values
        pred = self.model(input_d, age_d)

        # Use Tensorboard SummaryWriter to save and visualise model.
        # add to 'runs' folder with run specific name as prefix
        model_writer = SummaryWriter(os.path.join('runs', CONFIG.cli_args.tb_prefix, 'model'))
        model_writer.add_graph(self.model, (input_d, age_d))
        model_writer.close()

        print(f'label: {label_d.shape}, {label_d}')
        # use SmoothL1Loss for MAE
        loss_func = nn.MSELoss()
        loss = loss_func(
            pred,
            label_d
        )

        return loss.mean()

    # function to instantiate SummaryWriter for train and Validation loss
    def logMetrics(self, mode_str, epoch_ndx, loss_value):
        # mode_str: 'trn' or 'val'
        # loss_value: gets mean loss for the epoch
        self.initialiseTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')
        # This path ensures the two modes are grouped together during display
        writer.add_scalar('Loss/' + mode_str, loss_value, epoch_ndx)

    def initialiseTensorboardWriters(self):
        if self.trn_writer is None:
            # specify path to save the tran and validation event files to.
            # Date is included to ensure that files are not overwritten by files of the same names
            log_dir = os.path.join('runs', CONFIG.cli_args.tb_prefix, self.time_str)
            # Write to instantiated train and validation events files
            self.trn_writer = SummaryWriter(log_dir=log_dir + '-trn' + CONFIG.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '-val' + CONFIG.cli_args.comment)

    def main(self):
        train_dl = self.initialiseTrainDataloader()
        val_dl = self.initialiseValDataloader()
        print('Starting ___________________________________________________')
        # Arrays that collect mean loss every epoch(For debugging purpose)
        train_loss = []
        val_loss = []

        for epoch_ndx in range(1, CONFIG.cli_args.epochs + 1):
            print("Epoch {} of {}, {}/{} batch of size {}*{} with Learning rate:{}:".format(
                epoch_ndx,
                CONFIG.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                CONFIG.cli_args.batch_size,
                (CONFIG.CUDA_DEVICE_COUNT if CONFIG.USE_CUDA else 1),
                self.optimizer.param_groups[0]['lr']
            ))

            t_loss = self.doTraining(epoch_ndx, train_dl)
            train_loss.append(float(t_loss))
            self.logMetrics('trn', t_loss, epoch_ndx)

            v_loss = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics('val', v_loss, epoch_ndx)
            val_loss.append(float(v_loss))

            self.scheduler.step()

        print(f'Train_loss,{train_loss}')
        print(f'Val_loss,{val_loss}')

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

        # Save model and weights at this path
        torch.save({
            'epoch': CONFIG.cli_args.epochs,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.model.state_dict()
        }, '{}/{}.pth'.format(CONFIG.SAVE_WEIGHTS_PATH, CONFIG.cli_args.comment))


if __name__ == '__main__':
    RegTraining.main()
