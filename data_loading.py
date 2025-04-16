class VideoFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, feature_dir, flag="Train"):
        super().__init__()
        self.flag = flag
        normal_labels = ['12', '13', '15']
        if flag == "Train":
            self.feature_files = sorted(glob.glob(os.path.join(feature_dir, "*.npy")), key=lambda x: os.path.getsize(x))
        else:
            self.feature_files = glob.glob(os.path.join(feature_dir, "*.npy"))

        self.labels = [
            0 if f.split('_label_')[1].split('.')[0] in normal_labels else 1
            for f in self.feature_files]

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        features = np.load(self.feature_files[idx])
        label = self.labels[idx]

        if self.flag == "Train":
            if random.random() > 0.5:
                num_masks = random.randint(1, 2)
                for _ in range(num_masks):
                    mask_len = random.randint(2, features.shape[1] // 6)
                    start = random.randint(0, features.shape[1] - mask_len)
                    features[:, start:start + mask_len] = 0  
    
            if random.random() > 0.3:
                noise = np.random.normal(0, 0.02, features.shape)
                features += noise.astype(np.float32)

        if features.ndim == 1:
            features = features.reshape(3, -1)

        features = torch.from_numpy(features).float()
        return features, torch.tensor(label, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
