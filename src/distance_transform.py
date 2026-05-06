'''

Run only once for the data


csv_path = "/home/zobia/Boston_Datasets/MRI-data-20250805T065054Z-1-007/MRI-data/csv_files/NACC_train_80%.csv"

df = pd.read_csv(csv_path)

distance_dir = "/home/zobia/Boston_Datasets/NACC_distance_maps"
os.makedirs(distance_dir, exist_ok=True)

for path in tqdm(df["full_path"]):

    mri = np.load(path)

    mask = mri > 0
    dist = distance_transform_edt(mask)

    dist = dist / (dist.max() + 1e-8)

    filename = os.path.basename(path).replace(".npy", "_distance.npy")
    save_path = os.path.join(distance_dir, filename)

    np.save(save_path, dist)

print("Distance transforms created.")
'''
