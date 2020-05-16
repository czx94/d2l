from torch.utils.data import DataLoader


# TODO: finish the dataloader
class w2vDataloader(DataLoader):
    def __init__(self, ds, batch_size, num_workers=0):
        super().__init__(ds, batch_size, num_workers=num_workers)

    def __next__(self):
        pass