class Args:
    feature_dir = '/kaggle/input/ucf-crime-features'
    batch_size = 16
    lr = 0.001
    max_epoch = 20
    warmup = 5
    save_dir = './checkpoints'
    n_div = 8
    in_channels = 3
    out_channels = 512
    kernel_size = 8
    stride = 1

args = Args()
