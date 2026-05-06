from imports import *
from data_processing import *

# Load NACC_60%.csv for training
nacc_train_data = pd.read_csv('/home/zobia/Boston_Datasets/MRI-data-20250805T065054Z-1-007/MRI-data/csv_files/ADNI_ALL_test_100%.csv')
X_nacc_train = nacc_train_data['full_path'].values
y_nacc_train = nacc_train_data[['NC', 'MCI', 'AD']].values

# Load NACC_20%.csv for validation
nacc_val_data = pd.read_csv('//home/zobia/Boston_Datasets/MRI-data-20250805T065054Z-1-007/MRI-data/csv_files/ADNI_ALL_test_100%.csv')
X_nacc_val = nacc_val_data['full_path'].values
y_nacc_val = nacc_val_data[['NC', 'MCI', 'AD']].values

# Load ADNI_85%.csv for testing
adni_test_data = pd.read_csv('/home/zobia/Boston_Datasets/MRI-data-20250805T065054Z-1-007/MRI-data/csv_files/AIBL_test_100%.csv')
X_adni_test = adni_test_data['full_path'].values
y_adni_test = adni_test_data[['NC', 'MCI', 'AD']].values

# Total data set lengths
total_samples_train = len(X_nacc_train)
print(f"Total training data length: {total_samples_train}")
print(f"Total validation data length: {len(X_nacc_val)}")
print(f"Total testing data length: {len(X_adni_test)}")

# Class counts for the training dataset
nc_count_train = y_nacc_train[:, 0].sum()
mci_count_train = y_nacc_train[:, 1].sum()
ad_count_train = y_nacc_train[:, 2].sum()
print(f"Training dataset class counts: NC={nc_count_train}, MCI={mci_count_train}, AD={ad_count_train}")

# Class counts for validation dataset
nc_count_val = y_nacc_val[:, 0].sum()
mci_count_val = y_nacc_val[:, 1].sum()
ad_count_val = y_nacc_val[:, 2].sum()
print(f"Validation dataset class counts: NC={nc_count_val}, MCI={mci_count_val}, AD={ad_count_val}")

# Class counts for testing dataset
nc_count_test = y_adni_test[:, 0].sum()
mci_count_test = y_adni_test[:, 1].sum()
ad_count_test = y_adni_test[:, 2].sum()
print(f"Testing dataset class counts: NC={nc_count_test}, MCI={mci_count_test}, AD={ad_count_test}")

# Now create instances of the DataLoading for each split
train_dataset = DataLoading(
    X_nacc_train, 
    y_nacc_train, 
    distance_dir="/home/zobia/NACC/distance_transforms/", 
    transform=True, 
    augmentation=False, 
    use_distances=True  # Enable distance transforms for training
)

# Validation dataset (without distance transforms)
val_dataset = DataLoading(
    X_nacc_val, 
    y_nacc_val, 
    transform=False, 
    augmentation=False, 
    use_distances=False  # No distance transforms for validation
)

# Test dataset (without distance transforms)
test_dataset = DataLoading(
    X_adni_test, 
    y_adni_test, 
    transform=False, 
    augmentation=False, 
    use_distances=False  # No distance transforms for testing
)

# Calculate class weights for training dataset
class_weights = {
    0: total_samples_train / nc_count_train,
    1: total_samples_train / mci_count_train,
    2: total_samples_train / ad_count_train
}
print(f"Class weights for training dataset: {class_weights}")
# weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float)