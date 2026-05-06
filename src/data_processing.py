from imports import *

batch_size = 2
seed = 42
pl.seed_everything(seed = 42)

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

# Function to normalize the data
def minmax_normalize(data):
    return (data - data.min()) / (data.max() - data.min() + 0.0005)


class DataLoading(Dataset):
    def __init__(self, file_paths, labels, distance_dir=None, transform=True, augmentation=True, use_distances=False):
        self.file_paths = file_paths
        self.labels = labels
        self.distance_dir = distance_dir
        self.transform = transform
        self.use_distances = use_distances  # Toggle for loading distance transforms

        # Define transformations
        if transform and augmentation:
            self.transforms = tio.Compose([
                transforms.RandomApply(
                    [monai_transforms.RandSpatialCrop(
                        roi_size=(182//4, 218//4, 182//4),
                        random_center=True,
                        random_size=True
                    ),
                    monai_transforms.Resize(
                        spatial_size=(182, 218, 182)
                    )], p=0.5
                ),
                tio.RandomGamma(p=0.5),
                tio.RandomBiasField(p=0.25),
                tio.Lambda(minmax_normalize)
            ])
        elif transform and not augmentation:
            self.transforms = tio.Compose([
                tio.RandomGamma(p=0.5),
                tio.RandomBiasField(p=0.25),
                tio.Lambda(minmax_normalize)
            ])
        else:
            self.transforms = tio.Compose([
                tio.Lambda(minmax_normalize)
            ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load MRI data
        data = np.load(file_path)
        data = np.expand_dims(data, axis=0)  # Add channel dimension
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Optionally load corresponding distance transform
        distance_tensor = None
        if self.use_distances and self.distance_dir:
            file_name = os.path.basename(file_path).replace('.npy', '_distance.npy')
            distance_path = os.path.join(self.distance_dir, file_name)
            distance_transform = np.load(distance_path)
            distance_tensor = torch.tensor(distance_transform, dtype=torch.float32)

        # Apply transformations (to MRI data only)
        if self.transform:
            data_tensor = self.transforms(data_tensor)

        # Convert label to PyTorch tensor
        label_tensor = torch.tensor(label, dtype=torch.float)

        if self.use_distances:
            return data_tensor, label_tensor, distance_tensor
        return data_tensor, label_tensor


class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size, distributed_sampler=False):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.distributed_sampler = distributed_sampler

    def _get_data_loader(self, dataset, shuffle):
        sampler = DistributedSampler(dataset) if self.distributed_sampler else None

        def collate_fn(batch):
            if len(batch[0]) == 3:  # If distance transforms are included
                data_batch, label_batch, distance_batch = zip(*batch)
        
                # Process MRI data
                resized_data_batch = []
                for data in data_batch:
                    if data.dim() == 4:  # (C, X, Y, Z)
                        data = data.unsqueeze(0)  # Add batch dimension: (1, C, X, Y, Z)
                        data = F.interpolate(data, size=(176, 208, 176), mode='trilinear', align_corners=False)
                        data = data.squeeze(0)  # Remove batch dimension: (C, X', Y', Z')
                    elif data.dim() == 5:  # (B, C, X, Y, Z)
                        data = F.interpolate(data, size=(176, 208, 176), mode='trilinear', align_corners=False)
                    resized_data_batch.append(data)
        
                resized_data_batch = torch.stack(resized_data_batch)
                label_batch = torch.stack(label_batch)
                distance_batch = torch.stack(distance_batch)  # Stack distance transforms
        
                return resized_data_batch, label_batch, distance_batch
        
            else:  # Without distance transforms
                data_batch, label_batch = zip(*batch)
        
                # Process MRI data
                resized_data_batch = []
                for data in data_batch:
                    if data.dim() == 4:  # (C, X, Y, Z)
                        data = data.unsqueeze(0)  # Add batch dimension: (1, C, X, Y, Z)
                        data = F.interpolate(data, size=(176, 208, 176), mode='trilinear', align_corners=False)
                        data = data.squeeze(0)  # Remove batch dimension: (C, X', Y', Z')
                    elif data.dim() == 5:  # (B, C, X, Y, Z)
                        data = F.interpolate(data, size=(176, 208, 176), mode='trilinear', align_corners=False)
                    resized_data_batch.append(data)
        
                resized_data_batch = torch.stack(resized_data_batch)
                label_batch = torch.stack(label_batch)
        
                return resized_data_batch, label_batch

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler, num_workers=15, collate_fn=collate_fn,worker_init_fn=worker_init_fn)

    def train_dataloader(self):
        return self._get_data_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_data_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._get_data_loader(self.test_dataset, shuffle=False)