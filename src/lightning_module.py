from imports import *
from data_loading import DataLoading
from utilities import *
from classifier import TargetNet
from Unet3d import base_model

class LightningModule(pl.LightningModule):
    def __init__(self, model, num_classes, class_weights):
        super(LightningModule, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.test_step_outputs = []
        self.train_step_outputs = []
        self.val_step_outputs = []  

        # Weighted cross entropy
        weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float)
        self.criterion = SoftCrossEntropyLoss(weights=weights_tensor)
            
    def forward(self, x):
        return self.model(x)
    
    def calculate_and_log_metrics(self, y_true, y_pred, stage):

        metrics = compute_metrics(y_true, y_pred)

        self.log(f'{stage}_Overall_Accuracy', metrics['overall_accuracy'], on_epoch=True, prog_bar=True)
        self.log(f'{stage}_Macro_Precision', metrics['macro_precision'], on_epoch=True, prog_bar=True)
        self.log(f'{stage}_Macro_Recall', metrics['macro_recall'], on_epoch=True, prog_bar=True)
        self.log(f'{stage}_f1_macro', metrics['f1_macro'], on_epoch=True, prog_bar=True)



    def visualize_mri(self, mri_array, slice_index=None):
        """
        Visualize axial, sagittal, and coronal slices of a 3D MRI array.
    
        Args:
            mri_array (torch.Tensor or numpy.ndarray): A 3D MRI array to visualize.
            slice_index (tuple of int or None): The indices of the slices to visualize. 
                                                If None, defaults to the middle slices.
        """
        # If the input is a PyTorch tensor, convert it to a NumPy array
        if isinstance(mri_array, torch.Tensor):
            mri_array = mri_array.cpu().numpy()
    
        # Remove channel dimension if present
        if mri_array.ndim == 4 and mri_array.shape[0] == 1:
            mri_array = mri_array.squeeze(0)
    
        assert mri_array.ndim == 3, "Input array must be 3D."
    
        z, y, x = mri_array.shape
    
        # Default to middle slices if no slice index is provided
        if slice_index is None:
            slice_index = (z // 2, y // 2, x // 2)
    
        # Extract slices
        axial_slice = mri_array[slice_index[0], :, :]
        sagittal_slice = mri_array[:, slice_index[1], :]
        coronal_slice = mri_array[:, :, slice_index[2]]
    
        # Plot slices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(axial_slice, cmap='gray')
        axes[0].set_title('Axial View')
        axes[1].imshow(sagittal_slice, cmap='gray')
        axes[1].set_title('Sagittal View')
        axes[2].imshow(coronal_slice, cmap='gray')
        axes[2].set_title('Coronal View')
    
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def random_thresholds(self, min_val, max_val, min_fraction=0.1):
        """
        Generate thresholds for segmentation with guaranteed minimum region sizes.
        
        Args:
        - min_val (float): Minimum value of the distance transform.
        - max_val (float): Maximum value of the distance transform.
        - min_fraction (float): Minimum fraction of the range for each region.
        
        Returns:
        - threshold1 (float): First threshold.
        - threshold2 (float): Second threshold.
        """
        # Convert to CPU and scalar if needed
        if torch.is_tensor(min_val):
            min_val = min_val.cpu().item()
        if torch.is_tensor(max_val):
            max_val = max_val.cpu().item()
        
        # Range of the distance transform
        range_val = max_val - min_val
        min_region_size = range_val * min_fraction
    
        # Ensure thresholds have enough range for each region
        threshold1 = min_val + min_region_size
        threshold2 = max_val - min_region_size
    
        # Introduce randomness within constraints
        threshold1 = np.random.uniform(threshold1, (min_val + max_val) / 2)
        threshold2 = np.random.uniform((min_val + max_val) / 2, threshold2)
    
        return threshold1, threshold2

    
    def apply_distance_transform_mixup(self, x, distance_transforms, y):
        """
        Apply distance transform-based mixup with alpha and lambda parameters.
        Ensures that for a batch size of 2, the first image is mixed with the second and vice versa.
        """
        batch_size = x.size(0)
        assert batch_size == 2, "This implementation is specific to a batch size of 2."
                        
        # Explicit index for swapping (specific to batch size of 2)
        #index = torch.tensor([1, 0]).to(x.device)  # First swaps with second, second swaps with first
        index = torch.randperm(batch_size).to(x.device)

        
        # Extract samples and distance transforms
        x_a, x_b = x, x[index, :]
        dist_a, dist_b = distance_transforms, distance_transforms[index, :]
        y_a, y_b = y, y[index]
        
        # Generate random thresholds for distance-based segmentation
        threshold1_a, threshold2_a = self.random_thresholds(dist_a.min(), dist_a.max(), min_fraction=0.1)
        threshold1_b, threshold2_b = self.random_thresholds(dist_b.min(), dist_b.max(), min_fraction=0.1)

        # Compute regions
        # Compute regions sequentially to ensure non-overlapping
        region1_a = (dist_a <= threshold1_a).float()
        region2_b = ((dist_b > threshold1_a) & (dist_b <= threshold2_b) & (region1_a == 0)).float()
        region3_a = ((dist_a > threshold1_a) & (dist_a <= threshold2_a) & (region1_a == 0) & (region2_b == 0)).float()
        region4_b = ((dist_b > threshold2_b) & (region1_a == 0) & (region2_b == 0) & (region3_a == 0)).float()

        region1_a = region1_a.unsqueeze(1)
        region2_b = region2_b.unsqueeze(1)
        region3_a = region3_a.unsqueeze(1)
        region4_b = region4_b.unsqueeze(1)


        mixed_x = (
            (region1_a * x_a) +
            (region2_b * x_b) +
            (region3_a * x_a) +
            (region4_b * x_b)
        )

        #print(f"Image A - Min: {dist_a.min()}, Max: {dist_a.max()}, Thresholds: {threshold1_a}, {threshold2_a}")
        #print(f"Image B - Min: {dist_b.min()}, Max: {dist_b.max()}, Thresholds: {threshold1_b}, {threshold2_b}")
        #print("[Visualization] Region 1 (A)")
        #self.visualize_mri(region1_a[0] * x_a[0])  # Region 1 from Image A
        #print("[Visualization] Region 2 (B)")
        #self.visualize_mri(region2_b[0] * x_b[0])  # Region 2 from Image B
        #print("[Visualization] Region 3 (A)")
        #self.visualize_mri(region3_a[0] * x_a[0])  # Region 3 from Image A
        #print("[Visualization] Region 4 (B)")
        #self.visualize_mri(region4_b[0] * x_b[0])  # Region 4 from Image B


        #Visualize original images
        #print("[Visualization] Original Image A")
        #self.visualize_mri(x_a[0])  # Visualize the first sample in the batch
        #print("[Visualization] Original Image B")
        #self.visualize_mri(x_b[0])  # Visualize the second sample in the batch
        
        # Visualize mixed image
        #print("[Visualization] Mixed Image")
        #self.visualize_mri(mixed_x[0])  # Visualize the first mixed image
        #self.visualize_mri(mixed_x[1])
        
        #print(f"[apply_distance_transform_mixup] mixed_x shape: {mixed_x.shape}")
        
        # Compute contribution proportions
        pixels_a = (region1_a + region3_a).sum()
        pixels_b = (region2_b + region4_b).sum()
        total_pixels = pixels_a + pixels_b
        
        # Proportional mixing of labels
        prop_a = pixels_a / total_pixels
        prop_b = pixels_b / total_pixels
        
        # Mixed labels
        mixed_y = prop_a * y_a + prop_b * y_b
                
        #print(f"[apply_distance_transform_mixup] mixed_y shape: {mixed_y.shape}, mixed_y: {mixed_y}")
        
        return mixed_x, mixed_y
    
        
    def training_step(self, batch, batch_idx):
        """
        Training step for one batch with debug prints.
        Args:
        - batch: Tuple containing input images (x), labels (y), and distance transforms.
        - batch_idx: Index of the batch.
        """
        
        # Unpack batch
        x, y, distance_transforms = batch
        mixed_x, mixed_y= self.apply_distance_transform_mixup(x, distance_transforms, y)
      
        output = self.model(mixed_x)
       
        loss = self.criterion(output, mixed_y)  # mixed_y is in soft label format
      
        pred_labels = torch.argmax(output, dim=1)      
        y_hard = torch.argmax(y, dim=1)  # Original hard labels for comparison        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_step_outputs.append({'y_true': y_hard, 'y_pred': pred_labels})
        
        return loss



    def on_train_epoch_end(self):

        y_true = torch.cat([x['y_true'] for x in self.train_step_outputs])
        y_pred = torch.cat([x['y_pred'] for x in self.train_step_outputs])

        self.calculate_and_log_metrics(y_true, y_pred, 'train')
        
        self.train_step_outputs = []

    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch with debug prints.
        Args:
        - batch: Tuple containing input images (x) and labels (y).
        - batch_idx: Index of the batch.
        """
        
        # Unpack batch
        x, y = batch
    
        # Forward pass through the model
        output = self.model(x)
        y_true = torch.argmax(y, dim=1)

        val_loss = self.criterion(output, y)

        pred_labels = torch.argmax(output, dim=1)
    
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_step_outputs.append({'y_true': y_true, 'y_pred': pred_labels})
        
        return val_loss


    def on_validation_epoch_end(self):
        y_true = torch.cat([x['y_true'] for x in self.val_step_outputs])
        y_pred = torch.cat([x['y_pred'] for x in self.val_step_outputs])
    
        self.calculate_and_log_metrics(y_true, y_pred, 'val')
        
        self.val_step_outputs = []

    def test_step(self, batch, batch_idx):
        x, y = batch

        with torch.no_grad():
            output = self.model(x)
            y_indices = torch.argmax(y, dim=1)
            pred_labels = torch.argmax(output, dim=1)

            self.test_step_outputs.append({'y_true': y_indices, 'y_pred': pred_labels})

    def on_test_epoch_end(self):
        # Concatenate all the predictions and true labels from the test step outputs
        y = torch.cat([x['y_true'] for x in self.test_step_outputs])
        output_pred = torch.cat([x['y_pred'] for x in self.test_step_outputs])

        self.calculate_and_log_metrics(y, output_pred, 'test')

        # Calculate Cohen's kappa
        kappa = tmf.cohen_kappa(output_pred, y, task='multiclass', num_classes=self.num_classes)

        # Convert to numpy arrays for metric calculations
        y_np = y.cpu().numpy()
        output_pred_np = output_pred.cpu().numpy()
    
        # Calculate confusion matrix
        confusion_mat = confusion_matrix(y_np, output_pred_np, labels=[0, 1, 2])
        
        # Calculate MCC for each class
        num_classes = confusion_mat.shape[0]
        MCC = np.zeros(num_classes)
        for i in range(num_classes):
            TP = confusion_mat[i, i]
            FP = confusion_mat[:, i].sum() - TP
            FN = confusion_mat[i, :].sum() - TP
            TN = confusion_mat.sum() - (TP + FP + FN)
            
            MCC[i] = (TP * TN - FP * FN) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + 1e-9)
        
        # Calculate MACRO_MCC and WEIGHTED_MCC
        MACRO_MCC = MCC.mean()
        WEIGHTED_MCC = (MCC * confusion_mat.sum(axis=0)).sum() / confusion_mat.sum()
        
        # Log Cohen's kappa value
        self.log('test_kappa', kappa, prog_bar=True, sync_dist=True)
        self.log('test_macro_mcc', MACRO_MCC, prog_bar=True, sync_dist=True)
        self.log('test_weighted_mcc', WEIGHTED_MCC, prog_bar=True, sync_dist=True)
        
        # Plot confusion matrix
        df_cm = pd.DataFrame(confusion_mat, index=['NC', 'MCI', 'AD'], columns=['NC', 'MCI', 'AD'])
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap='Spectral', fmt='g')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
        
        # Clear the outputs for the next epoch
        self.test_step_outputs = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=False)
        exp_lambda: float = 0.95
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: exp_lambda ** epoch)

        return [optimizer], [scheduler]

# Load pre-trained weights
# -- Remove module keyword from state dict since it pre-trained model was trained on dataparallel
weight_dir = '/home/zobia/MedicalNet/pretrain/Genesis_Chest_CT.pt'
checkpoint = torch.load(weight_dir, map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
base_model.load_state_dict(unParalled_state_dict)

# Initialize target model after loading weights
# -- Target model is the classification model
# -- the base model is originally for segmentation that checkpoint is trained on
num_classes = 3
target_model = TargetNet(base_model, num_classes)