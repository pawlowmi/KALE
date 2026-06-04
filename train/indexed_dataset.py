from torchvision.datasets import ImageFolder


class IndexedImageFolder(ImageFolder):
    """ImageFolder that also returns the sample index for precomputed embedding lookup."""
    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        return sample, target, idx
