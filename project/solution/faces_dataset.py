"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = [os.path.join(self.root_path, 'real',name) for name in os.listdir(os.path.join(self.root_path, 'real'))]
        
        self.fake_image_names = [os.path.join(self.root_path, 'fake',name) for name in os.listdir(os.path.join(self.root_path, 'fake'))]
        
        entire_pathes = self.real_image_names + self.fake_image_names
        self.images_with_labels = [(path, 0 if 'real' in path.split('/') else 1) for path in entire_pathes]

        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        
        
        image_path, label = self.images_with_labels[index]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """Return the number of images in the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        
        return len(self.images_with_labels)
