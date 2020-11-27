from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from utils import data_utils


class PerceptConceptDataset(ImageFolder):
    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
        super(PerceptConceptDataset, self).__init__(source_root, source_transform)
        self.target_image_paths = sorted(data_utils.make_dataset(target_root))
        self.target_image_transform = target_transform
        self.opts = opts

    def __getitem__(self, index):
        from_im, target = ImageFolder.__getitem__(self, index)
        to_path = self.target_image_paths[index]
        to_im = Image.open(to_path).convert('RGB')
        if self.target_image_transform:
            to_im = self.target_image_transform(to_im)

        return from_im, to_im, target


if __name__ == "__main__":
    path = r''
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    ds = PerceptConceptDataset(path, path, None, t, t)
    dl = DataLoader(ds)

    former = None
    for source, target, label in dl:
        print(source.data == target.data)
        print(label)
        former = source
