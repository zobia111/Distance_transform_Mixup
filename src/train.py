from imports import *

from Unet3d import UNet3D
from classifier import TargetNet

from utilities import *

from data_loading import *
from lightning_module import *

from lightning_module import LightningModule

wandb_logger = WandbLogger(project='Resnet3D', log_model='all', save_dir='./logs')
data_module = DataModule(train_dataset, val_dataset, test_dataset, batch_size, distributed_sampler=False)

# Instantiate PyTorch Lightning Model
#lightning_model = LightningModule(target_model, criterion,num_classes, class_weights=class_weights)
lightning_model = LightningModule(model=target_model, num_classes=num_classes, class_weights=class_weights)

# Best model Checkpoint
checkpoint_callback = ModelCheckpoint(save_top_k=1, dirpath='./ResnetCheck/', 
                                  filename='Best_checkpoint-{epoch:02d}-{val_loss:.3f}',
                                      monitor='val_loss', mode='min')

 # Define early stopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=15,          # Number of epochs with no improvement after which training will be stopped
    mode='min'            # Whether to minimize or maximize the monitored metric ('min' for loss, 'max' for accuracy, etc.)
    )

# PyTorch Lightning Trainer
trainer = pl.Trainer(
    logger=wandb_logger,
    max_epochs=90,
    accelerator='gpu',  # 'ddp' accelerator is used for distributed data-parallel training
    precision='16-mixed' ,         # Enables mixed precision training (float16)
    #gradient_clip_val=1.0,
    num_nodes=1,
    devices=1,  # Set the number of GPUs
    #strategy='ddp_notebook',
    accumulate_grad_batches=8,
    #limit_train_batches= 200,
    callbacks=[
            checkpoint_callback,
            early_stopping_callback
            #GradientAccumulationScheduler(batch_size=[4, 8, 16])  # Optional: vary batch size dynamically
        ]
)

# Train the model
#trainer.fit(lightning_model, datamodule=data_module)
   
# Test the model
#best_checkpoint_path = checkpoint_callback.best_model_path
#print(f"Using checkpoint for testing: {best_checkpoint_path}")
#trainer.save_checkpoint(best_checkpoint_path)
#trainer.test(ckpt_path=best_checkpoint_path, datamodule=data_module)