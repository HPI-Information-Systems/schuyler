import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

class TUTALightningModule(pl.LightningModule):
    def __init__(self, model, config, margin, lr=1e-4):
        super().__init__()
        self.model = model
        self.config = config
        self.lr = lr
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    
    def forward(self, **inputs):
        return self.model(**inputs)

    # def forward(self, batch):
    #     token_id, num_mag, num_pre, num_top, num_low, \
    #     token_order, pos_row, pos_col, pos_top, pos_left, \
    #     format_vec, indicator = batch["inputs"]  #todo take as input

    #     # Forward pass with TUTA
    #     logits = self.model(
    #         token_id=token_id,
    #         num_mag=num_mag,
    #         num_pre=num_pre,
    #         num_top=num_top,
    #         num_low=num_low,
    #         token_order=token_order,
    #         pos_row=pos_row,
    #         pos_col=pos_col,
    #         pos_top=pos_top,
    #         pos_left=pos_left,
    #         format_vec=format_vec,
    #         indicator=indicator,
    #     ) #model is: TUTAForTriplet, which calls basemodel and then the finetuning head
    #     return logits

    def training_step(self, batch, batch_idx):        
        anchor_inputs = batch["anchor"]
        positive_inputs = batch["positive"]
        negative_inputs = batch["negative"]

        anchor_emb = self.model(**anchor_inputs)
        positive_emb = self.model(**positive_inputs)
        negative_emb = self.model(**negative_inputs)

        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        wandb.log({"train_loss": loss})
        return loss
    
    def training_step(self, batch, batch_idx):        
        anchor_inputs = batch["anchor"]
        positive_inputs = batch["positive"]
        negative_inputs = batch["negative"]

        anchor_emb = self.model(**anchor_inputs)
        positive_emb = self.model(**positive_inputs)
        negative_emb = self.model(**negative_inputs)

        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        wandb.log({"val_loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)