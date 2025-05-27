import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

class TUTALightningModule(pl.LightningModule):
    def __init__(self, model, margin, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    
    def forward(self, inputs):
        return self.model(*inputs)

    def training_step(self, batch, batch_idx):
        # print("batch", batch)
        # print("size", len(batch))
        anchor_inputs, positive_inputs, negative_inputs = batch
        anchor_emb = self.model(*anchor_inputs)
        positive_emb = self.model(*positive_inputs)
        negative_emb = self.model(*negative_inputs)

        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #wandb.log({"train_loss": loss})
        return loss
    
    def validation_step(self, batch, batch_idx):        
        # print("batch", batch)
        # print("size", len(batch))
        anchor_inputs, positive_inputs, negative_inputs = batch

        anchor_emb = self.model(*anchor_inputs)
        positive_emb = self.model(*positive_inputs)
        negative_emb = self.model(*negative_inputs)

        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)