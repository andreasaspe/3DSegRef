class BaseDataset:
    def __init__(self, data, transform=None):
        self.data = data          # this can be *anything iterable* (list, another dataset, etc.)
        self.transform = transform

    def __getitem__(self, index):
        # Get one item
        item = self.data[index]   # <-- this is the magic line
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.data)

class PersistentDataset(BaseDataset):
    def __getitem__(self, index):
        print(f"[PersistentDataset] Loading item {index} and preprocessing it once...")
        item = self.data[index]
        # imagine preprocessing happens here
        return f"preprocessed({item})"
    
# Let's pretend these are file paths
raw_files = ["case_0.nii.gz", "case_1.nii.gz", "case_2.nii.gz"]

# Step 1: PersistentDataset (preprocessing + caching)
train_ds = PersistentDataset(data=raw_files)

# Step 2: Wrap PersistentDataset with another Dataset for augmentation
def add_augmentation(x):
    return f"augmented({x})"

train_ds_with_aug = BaseDataset(data=train_ds, transform=add_augmentation)

print(train_ds_with_aug[0])
