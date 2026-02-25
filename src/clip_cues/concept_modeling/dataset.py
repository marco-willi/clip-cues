"""Dataset for concept modeling."""

from torch.utils.data import Dataset


class CLIPFeatureDataset(Dataset):
    """Dataset for CLIP features and labels.

    Args:
        clip_features: Tensor of CLIP features [N, D]
        labels: Tensor of labels [N]
        image_ids: List of image IDs
    """

    def __init__(self, clip_features, labels, image_ids: list[str]):
        self.clip_features = clip_features
        self.labels = labels
        self.image_ids = image_ids

    def __len__(self):
        return self.clip_features.shape[0]

    def __getitem__(self, idx):
        return self.clip_features[idx, :], self.labels[idx], self.image_ids[idx]
