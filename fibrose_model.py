import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchmetrics
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import os
class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes,model_name, learning_rate=3e-4):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes

        if model_name == "resnet":
            self.feature_extractor = models.resnet50(pretrained=True)
            self.feature_extractor.eval()
            layer4 = self.feature_extractor.layer4
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        

        elif model_name == "vgg":
            self.feature_extractor = models.vgg16_bn(pretrained=True)
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False


        n_sizes = self._get_conv_output(input_shape)

        self.classifier = nn.Linear(n_sizes, num_classes)
    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x
    
    # will be used during inference
    def forward(self, x):
       x = self._forward_features(x)
       x = x.view(x.size(0), -1)
       x = F.log_softmax(self.classifier(x))
       
       return x

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
       
        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,weight_decay=0.1)
        return optimizer
