
#@title UMAP utils
from pynndescent import NNDescent
import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
import torch

import torch
from torch import nn
import torch.nn.functional as F


#@title UMAP modules
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
import torch.nn.functional as F

from umap_pytorch.data import UMAPDataset, MatchDataset
from umap_pytorch.modules import get_umap_graph, umap_loss
from umap_pytorch.model import default_encoder, default_decoder

from umap.umap_ import find_ab_params
import dill
from umap import UMAP
from pynndescent import NNDescent
import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
import torch

def convert_distance_to_probability(distances, a=1.0, b=1.0):
    return -torch.log1p(a * distances ** (2 * b))

def compute_cross_entropy(
    probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=3.0
):
    # cross entropy
    attraction_term = -probabilities_graph * torch.nn.functional.logsigmoid(
        probabilities_distance
    )
    repellant_term = (
        -(1.0 - probabilities_graph)
        * (torch.nn.functional.logsigmoid(probabilities_distance)-probabilities_distance)
        * repulsion_strength)

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE

def umap_loss(embedding_to, embedding_from, _a, _b, batch_size, negative_sample_rate=5):
    # get negative samples by randomly shuffling the batch
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
    distance_embedding = torch.cat((
        (embedding_to - embedding_from).norm(dim=1),
        (embedding_neg_to - embedding_neg_from).norm(dim=1)
    ), dim=0)

    # convert probabilities to distances
    probabilities_distance = convert_distance_to_probability(
        distance_embedding, _a, _b
    )
    # set true probabilities based on negative sampling
    probabilities_graph = torch.cat(
        (torch.ones(batch_size), torch.zeros(batch_size * negative_sample_rate)), dim=0,
    )

    # compute cross entropy
    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph.cuda(),
        probabilities_distance.cuda(),
    )
    loss = torch.mean(ce_loss)
    return loss

def get_umap_graph(X, n_neighbors=10, metric="cosine", random_state=None):
    random_state = check_random_state(None) if random_state == None else random_state
    # number of trees in random projection forest
    n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(X.shape[0]))))
    # distance metric

    # get nearest neighbors
    nnd = NNDescent(
        X,#.reshape((len(X), np.product(np.shape(X)[1:]))),
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph

    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph
    # build fuzzy_simplicial_set
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X = X,
        n_neighbors = n_neighbors,
        metric = metric,
        random_state = random_state,
        knn_indices= knn_indices,
        knn_dists = knn_dists,
    )
    
    return umap_graph

""" Model """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl

# # Assuming umap_loss and find_ab_params are defined elsewhere
# from umap.parametric_umap import umap_loss, find_ab_params

# class Model(pl.LightningModule):
#     def __init__(
#         self,
#         lr: float,
#         encoder: nn.Module,
#         decoder: nn.Module, # Decoder is required for fine-tuning
#         beta: float = 1.0,
#         gamma: float = 1.0, # NEW: Weight for the cyclic loss
#         fine_tune_decoder: bool = False, # NEW: Flag to enable fine-tuning mode
#         min_dist=0.1,
#         reconstruction_loss=F.binary_cross_entropy_with_logits,
#         match_nonparametric_umap=False,
#     ):
#         super().__init__()
#         # Using save_hyperparameters is good practice in PyTorch Lightning
#         self.save_hyperparameters(ignore=['encoder', 'decoder', 'reconstruction_loss'])
#         self.encoder = encoder
#         self.decoder = decoder
#         self.reconstruction_loss = reconstruction_loss
#         self._a, self._b = find_ab_params(1.0, min_dist)

#         # ✅ 1. Freeze the encoder if we are in fine-tuning mode
#         if self.hparams.fine_tune_decoder:
#             print("✅ Fine-tuning mode enabled: Encoder weights are frozen.")
#             for param in self.encoder.parameters():
#                 param.requires_grad = False

#     def configure_optimizers(self):
#         # ✅ 2. Only pass the decoder's parameters to the optimizer
#         if self.hparams.fine_tune_decoder:
#             print("✅ Optimizer configured to update decoder weights only.")
#             return torch.optim.AdamW(self.decoder.parameters(), lr=self.hparams.lr)
#         else:
#             # Original behavior: train all parameters
#             return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

#     def training_step(self, batch, batch_idx):
#         # Route to the correct training logic based on the mode
#         if self.hparams.fine_tune_decoder:
#             return self.decoder_finetuning_step(batch)
        
#         # Original training logic is preserved below
#         if not self.hparams.match_nonparametric_umap:
#             (edges_to_exp, edges_from_exp) = batch
#             embedding_to, embedding_from = self.encoder(edges_to_exp), self.encoder(edges_from_exp)
#             encoder_loss = umap_loss(embedding_to, embedding_from, self._a, self._b, edges_to_exp.shape[0], negative_sample_rate=5)
#             self.log("umap_loss", encoder_loss, prog_bar=True)
            
#             if self.decoder:
#                 recon = self.decoder(embedding_to)
#                 recon_loss = self.reconstruction_loss(recon, edges_to_exp)
#                 self.log("recon_loss", recon_loss, prog_bar=True)
#                 return encoder_loss + self.hparams.beta * recon_loss
#             else:
#                 return encoder_loss
            
#         else:
#             data, embedding = batch
#             embedding_parametric = self.encoder(data)
#             encoder_loss = F.mse_loss(embedding_parametric, embedding)
#             self.log("encoder_loss", encoder_loss, prog_bar=True)
            
#             if self.decoder:
#                 recon = self.decoder(embedding_parametric)
#                 recon_loss = self.reconstruction_loss(recon, data)
#                 self.log("recon_loss", recon_loss, prog_bar=True)
#                 return encoder_loss + self.hparams.beta * recon_loss
#             else:
#                 return encoder_loss

#     # ✅ 3. New training step for fine-tuning with cyclic loss
#     def decoder_finetuning_step(self, batch):
#         # This assumes the dataloader provides the original data and the target UMAP embedding
#         data, embedding = batch

#         # Loss 1: Reconstruction Loss (main decoder task)
#         # Reconstruct the original data from its embedding
#         recon = self.decoder(embedding)
#         recon_loss = self.reconstruction_loss(recon, data)
#         self.log("ft_recon_loss", recon_loss, prog_bar=True)

#         # Loss 2: Cyclic Loss (consistency task)
#         # Pass the reconstruction back through the frozen encoder and check
#         # if the resulting embedding matches the original embedding.
#         re_encoded_embedding = self.encoder(recon)
#         cyclic_loss = F.mse_loss(embedding, re_encoded_embedding)
#         self.log("ft_cyclic_loss", cyclic_loss, prog_bar=True)

#         # Total loss is a weighted sum of the two
#         total_loss = self.hparams.beta * recon_loss + self.hparams.gamma * cyclic_loss
#         self.log("ft_total_loss", total_loss)
#         return total_loss

class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        encoder: nn.Module,
        decoder=None,
        beta = 1.0,
        min_dist=0.1,
        reconstruction_loss=F.binary_cross_entropy_with_logits,
        match_nonparametric_umap=False,
    ):
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta # weight for reconstruction loss
        self.match_nonparametric_umap = match_nonparametric_umap
        self.reconstruction_loss = reconstruction_loss
        self._a, self._b = find_ab_params(1.0, min_dist)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        if not self.match_nonparametric_umap:
            (edges_to_exp, edges_from_exp) = batch
            embedding_to, embedding_from = self.encoder(edges_to_exp), self.encoder(edges_from_exp)
            encoder_loss = umap_loss(embedding_to, embedding_from, self._a, self._b, edges_to_exp.shape[0], negative_sample_rate=5)
            self.log("umap_loss", encoder_loss, prog_bar=True)
            
            if self.decoder:
                recon = self.decoder(embedding_to)
                recon_loss = self.reconstruction_loss(recon, edges_to_exp)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss
            else:
                return encoder_loss
            
        else:
            data, embedding = batch
            embedding_parametric = self.encoder(data)
            encoder_loss = mse_loss(embedding_parametric, embedding)
            self.log("encoder_loss", encoder_loss, prog_bar=True)
            if self.decoder:
                recon = self.decoder(embedding_parametric)
                recon_loss = self.reconstruction_loss(recon, data)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss
            else:
                return encoder_loss
            

""" Datamodule """


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size,        
        num_workers,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

class PUMAP():
    def __init__(
        self,
        encoder=None,
        decoder=None,
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        n_components=2,
        beta=1.0,
        reconstruction_loss=F.binary_cross_entropy_with_logits,
        random_state=None,
        lr=1e-3,
        epochs=10,
        batch_size=64,
        num_workers=1,
        num_gpus=1,
        match_nonparametric_umap=False,
        nonparametric_embeddings=None,
        save_ckpts=False
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.match_nonparametric_umap = match_nonparametric_umap
        self.nonparametric_embeddings = nonparametric_embeddings
        self.save_ckpts = save_ckpts
        
    def fit(self, X):
        if self.save_ckpts:
            import os
            from pytorch_lightning.callbacks import ModelCheckpoint

            from pytorch_lightning.callbacks import Callback
            class EveryNEpochsCheckpoint(Callback):
                """
                Save checkpoints every N epochs without blocking training.
                
                Args:
                    save_dir: Directory to save checkpoints
                    every_n_epochs: Save checkpoint every N epochs (default: 5)
                    filename: Filename pattern (can use {epoch} placeholder)
                """
                
                def __init__(
                    self,
                    save_dir: str = "checkpoints",
                    every_n_epochs: int = 5,
                    filename: str = "checkpoint_epoch_{epoch:03d}.ckpt"
                ):
                    self.save_dir = save_dir
                    self.every_n_epochs = every_n_epochs
                    self.filename = filename
                    os.makedirs(save_dir, exist_ok=True)
                
                def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
                    """Save checkpoint at the end of every N epochs"""
                    epoch = trainer.current_epoch
                    
                    # Check if this epoch should trigger a save
                    if (epoch + 1) % self.every_n_epochs == 0:
                        filepath = os.path.join(
                            self.save_dir,
                            self.filename.format(epoch=epoch)
                        )
                        # Use trainer's save_checkpoint which is async and non-blocking
                        trainer.save_checkpoint(filepath)
                        print(f"Saved checkpoint: {filepath}")

            # Create the callback
            checkpoint_callback = EveryNEpochsCheckpoint(
                save_dir="my_checkpoints",
                every_n_epochs=5,
                filename="model_epoch_{epoch:03d}.ckpt",
                # monitor='val_loss',          # The metric to monitor
            )

            # checkpoint_callback = ModelCheckpoint(
            #     dirpath='/colab/glass-box-umap/ckpts_v2/',
            #     filename='epoch={epoch:02d}-val_loss={val_loss:.2f}',
            #     monitor='val_loss',
            #     save_top_k=1,  # Save all checkpoints
            #     every_n_epochs=4  # Save checkpoint every n epochs
            # )
            # Configure the callback to save the best model based on validation loss
            # checkpoint_callback = ModelCheckpoint(
            #     dirpath='/colab/glass-box-umap/ckpts_v2/',  # Directory to save checkpoints
            #     filename='best-model-{epoch:02d}-{val_loss:.2f}', # Dynamic filename
            #     monitor='val_loss',          # The metric to monitor
            #     mode='min',                  # 'min' for loss, 'max' for accuracy
            #     save_top_k=1,                # Save only the single best model
            #     verbose=True                 # Optional: print a message when a checkpoint is saved
            # )

            trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs, callbacks=[checkpoint_callback])#, default_root_dir='/colab/glass-box-umap/ckpts_1007')#
            # print(checkpoint_callback)
            # print('ckpts')
            # trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs, default_root_dir='/colab/glass-box-umap/ckpts_1007')
        else:
            trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs)
        encoder = default_encoder(X.shape[1:], self.n_components) if self.encoder is None else self.encoder
        
        if self.decoder is None or isinstance(self.decoder, nn.Module):
            decoder = self.decoder
        elif self.decoder == True:
            decoder = default_decoder(X.shape[1:], self.n_components)
            
            
        if not self.match_nonparametric_umap:
            self.model = Model(self.lr, encoder, decoder, beta=self.beta, min_dist=self.min_dist, reconstruction_loss=self.reconstruction_loss)
            graph = get_umap_graph(X, n_neighbors=self.n_neighbors, metric=self.metric, random_state=self.random_state)
            trainer.fit(
                model=self.model,
                datamodule=Datamodule(UMAPDataset(X, graph), self.batch_size, self.num_workers)
                )
            if self.save_ckpts:
                print(checkpoint_callback.best_model_path)
        else:
            if self.nonparametric_embeddings is None:
                print("Fitting Non parametric Umap")
                non_parametric_umap = UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, metric=self.metric, n_components=self.n_components, random_state=self.random_state, verbose=True)
                self.non_parametric_embeddings = non_parametric_umap.fit_transform(torch.flatten(X, 1, -1).numpy())
            self.model = Model(self.lr, encoder, decoder, beta=self.beta, reconstruction_loss=self.reconstruction_loss, match_nonparametric_umap=self.match_nonparametric_umap)
            print("Training NN to match embeddings")
            trainer.fit(
                model=self.model,
                datamodule=Datamodule(MatchDataset(X, self.nonparametric_embeddings), self.batch_size, self.num_workers)
            )
        
    @torch.no_grad()
    def transform(self, X):
        return self.model.encoder(X).detach().cpu().numpy()
    
    @torch.no_grad()
    def inverse_transform(self, Z):
        return self.model.decoder(Z).detach().cpu().numpy()
    
    def save(self, path):
        with open(path, 'wb') as oup:
            dill.dump(self, oup)
        print(f"Pickled PUMAP object at {path}")
        
def load_pumap(path): 
    print("Loading PUMAP object from pickled file.")
    with open(path, 'rb') as inp: return dill.load(inp)
