from torch.utils.data import Dataset, DataLoader
from schuyler.solutions.tuta.prepare import main as prepare
from convert_df_to_jsonl import write_to_jsonl, dataframe_to_wikitables_json
import schuyler.solutions.tuta.dynamic_data as dymdata
#tables need to be sorted by alphabetical order. I need to proof this!

#convert to json files
#-> convert_df_to_jsonl.py
#apply prepare function
table = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
write_to_jsonl(dataframe_to_wikitables_json(table, 1, "Test"), "table.jsonl")

# put this data into dynamic dataloader
#loop through the dataset and add them to the triplet dataset


class TripletDataset(Dataset):
    def __init__(self, data_triplets, transform=None):
        super().__init__()
        self.data_triplets = data_triplets
        self.transform = transform

    def __len__(self):
        return len(self.data_triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.data_triplets[idx]

        # mlm_triple, sep_triple, tok_triple, tcr_triple = model(
        #     token_id, num_mag, num_pre, num_top, num_low, 
        #     token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator, 
        #     mlm_label, clc_label, tcr_label
        # )
        return anchor, positive, negative