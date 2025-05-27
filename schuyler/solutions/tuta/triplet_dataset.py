from torch.utils.data import Dataset
class TripletDataset(Dataset):
    def __init__(self, triplets, table_dict):
        super().__init__()
        self.triplets = triplets
        self.table_dict = table_dict
        self.data_triplets = []
        print(self.table_dict.keys())
        for anc, pos, neg in self.triplets:
            self.data_triplets.append(
                (
                    self.table_dict[anc.table.table_name],
                    self.table_dict[pos.table.table_name],
                    self.table_dict[neg.table.table_name],
                )
            )

    def __len__(self):
        return len(self.data_triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.data_triplets[idx]
        return anchor, positive, negative