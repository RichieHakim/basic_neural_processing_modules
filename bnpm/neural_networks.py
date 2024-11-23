import copy

import torch
import numpy as np
from tqdm.auto import tqdm
import sklearn.neural_network

from . import indexing

class RegressionRNN(torch.nn.Module):
    """
    RNN for time series-like regression

    RH 2023

    Args:
        input_size (int):
            Number of features in the input.
        hidden_size (int):
            Number of features in the hidden state.
        num_layers (int):
            Number of recurrent layers.
        output_size (int):
            Number of features in the output.
        batch_first (bool):
            If True, then the input and output tensors are provided as (batch,
            seq, feature). Default: True
        architecture (str):
            * 'RNN': RNN
            * 'GRU': GRU
            * 'LSTM': LSTM
        kwargs_architecture (dict):
            Extra keyword arguments for the architecture.
        nonlinearity (str):
            Nonlinearity to use. Can be either 'tanh' or 'relu'.
            Only used if architecture is RNN.
        bias (bool):
            If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        dropout (float):
            If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
        bidirectional (bool):
            If True, becomes a bidirectional RNN. Default: False
        device (str):
            Device to use.
        dtype (torch.dtype):
            Data type to use.
    """
    def __init__(
        self, 
        features=1, 
        hidden_size=100, 
        num_layers=1, 
        output_size=1,
        batch_first=True,
        architecture='LSTM',
        kwargs_architecture={},
        nonlinearity='tanh',
        bias=True,
        dropout=0.0,
        bidirectional=False,
        device='cpu',
        dtype=torch.float32,
    ):
        super().__init__()
        self.input_size = features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first
        self.architecture = architecture.upper()
        self.kwargs_architecture = kwargs_architecture
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device
        self.dtype = dtype

        assert hasattr(torch.nn, self.architecture), f'Architecture {self.architecture} not implemented'

        if self.architecture in ['RNN']:
            self.kwargs_architecture['nonlinearity'] = self.nonlinearity

        self.rnn = getattr(torch.nn, self.architecture)(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=self.batch_first,
            device=self.device,
            dtype=self.dtype,
            bias=self.bias,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            **self.kwargs_architecture,
        )
        self.fc = torch.nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.output_size)

    def forward(self, x, hidden=None, hidden_initialization='zeros'):
        """
        Forward pass of the RNN

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, seq_len, input_size).
            hidden (torch.Tensor):
                Initial hidden state.
                If architecture is RNN or GRU: \
                    * Shape: (num_layers * (2 if bidirectional else 1), batch_size, hidden_size)
                If architecture is LSTM: \
                    * Shape: (num_layers * (2 if bidirectional else 1), batch_size, hidden_size)
                    * Tuple of (h0, c0)
            hidden_initialization (str):
                Initialization method used if hidden is None.
                    * 'zeros': Zeros
                    * 'ones': Ones
                    * 'random_normal': Random normal
                    * 'random_uniform': Random uniform
                    * 'xavier_normal': Xavier normal
                    * 'xavier_uniform': Xavier uniform
                    * 'kaiming_normal': Kaiming normal
                    * 'kaiming_uniform': Kaiming uniform
                    * 'orthogonal': Orthogonal
                                    
        Returns:
            out (torch.Tensor):
                Output tensor of shape (batch_size, seq_len, output_size).
        """
        if hidden is None:
            hidden = self.initialize_hidden(
                batch_size=x.size(0), 
                hidden_initialization=hidden_initialization,
            )
        out, hidden = self.rnn(x, hidden)  ## Note that hidden is a tuple for LSTM (h0, c0)
        out = self.fc(out)
        return out, hidden
    
    def initialize_hidden(self, batch_size, hidden_initialization='zeros'):
        """
        Initialize hidden state

        Args:
            batch_size (int):
                Batch size.
            hidden_initialization (str):
                Initialization method used if hidden is None.
                    * 'zeros': Zeros
                    * 'ones': Ones
                    * 'random_normal': Random normal
                    * 'random_uniform': Random uniform
                    * 'xavier_normal': Xavier normal
                    * 'xavier_uniform': Xavier uniform
                    * 'kaiming_normal': Kaiming normal
                    * 'kaiming_uniform': Kaiming uniform
                    * 'orthogonal': Orthogonal

        Returns:
            hidden (torch.Tensor):
                Initial hidden state.
                If architecture is RNN or GRU: \
                    * Shape: (num_layers * (2 if bidirectional else 1), batch_size, hidden_size)
                If architecture is LSTM: \
                    * Shape: (num_layers * (2 if bidirectional else 1), batch_size, hidden_size)
                    * Tuple of (h0, c0)
        """
        
        fn_empty = lambda : torch.empty(
            self.num_layers * (2 if self.rnn.bidirectional else 1),
            batch_size,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )

        ## Make lambda functions that fill an empty tensor with the desired initialization
        if hidden_initialization == 'zeros':
            fn_fill = lambda e: torch.nn.init.zeros_(e)
        elif hidden_initialization == 'ones':
            fn_fill = lambda e: torch.nn.init.ones_(e)            
        elif hidden_initialization == 'random_normal':
            fn_fill = lambda e: torch.nn.init.normal_(e)
        elif hidden_initialization == 'random_uniform':
            fn_fill = lambda e: torch.nn.init.uniform_(e)
        elif hidden_initialization == 'xavier_normal':
            fn_fill = lambda e: torch.nn.init.xavier_normal_(e)
        elif hidden_initialization == 'xavier_uniform':
            fn_fill = lambda e: torch.nn.init.xavier_uniform_(e)
        elif hidden_initialization == 'kaiming_normal':
            fn_fill = lambda e: torch.nn.init.kaiming_normal_(e)
        elif hidden_initialization == 'kaiming_uniform':
            fn_fill = lambda e: torch.nn.init.kaiming_uniform_(e)
        elif hidden_initialization == 'orthogonal':
            fn_fill = lambda e: torch.nn.init.orthogonal_(e)
        else:
            raise ValueError(f'Unknown hidden_initialization {hidden_initialization}')
        
        ## Fill the empty tensor
        if self.architecture in ['RNN', 'GRU']:
            hidden = fn_fill(fn_empty())
        elif self.architecture == 'LSTM':
            hidden = (fn_fill(fn_empty()), fn_fill(fn_empty()))
        
        return hidden
    
    def to(self, device):
        self.device = device
        self.rnn.to(device)
        self.fc.to(device)
        return self
    

class Dataloader_MultiTimeSeries():
    """
    Dataloader for loading random chunks of multiple time series that all have
    the same number of features but may have different durations.
    The inputs are a list of 2D arrays of shape (duration, num_features).
    The output is a batch of shape (batch_size=n_datasets, batch_duration,
    num_features).
    Note that this is not a true pytorch Dataloader class, and does not support
    multiprocessing and other features.

    Args:
        timeseries (list):
            List of 2D arrays of shape (duration, num_features).
            Each array should be a torch.Tensor
            If 1D, then it will be converted to 2D with shape (duration, 1).
        batch_size (int):
            Batch size.
        batch_duration (int):
            Duration of each batch.
        shuffle_datasets (bool):
            If True, then the datasets are shuffled.
            If False, then the batch_size will be the number of datasets and the
            batch_size argument will be ignored.
        shuffle_startIdx (bool):
            If True, then the starting indices of each batch are shuffled each epoch.
            If False, then the sequence will be sequential chunks of
            batch_duration starting at 0.
        device (str):
            Device to use.
        """
    def __init__(
        self, 
        timeseries, 
        batch_size=1, 
        batch_duration=100, 
        shuffle_datasets=True, 
        shuffle_startIdx=True,
    ): 
        if not isinstance(timeseries, list):
            timeseries = [timeseries]
        
        assert all([isinstance(ts, torch.Tensor) for ts in timeseries]), 'All timeseries must be torch.Tensor'
        timeseries = [ts[:,None] if ts.ndim == 1 else ts for ts in timeseries]
        assert all([ts.ndim == 2 for ts in timeseries]), 'All timeseries must be 2D'
        assert all([ts.shape[1] == timeseries[0].shape[1] for ts in timeseries]), f"All timeseries must have the same number of features. Got {[ts.shape for ts in timeseries]}"
        
        self.batch_size = batch_size
        self.batch_duration = batch_duration
        self.shuffle_datasets = shuffle_datasets
        self.shuffle_startIdx = shuffle_startIdx

        self.num_datasets = len(timeseries)
        self.num_features = timeseries[0].shape[1]
        self.durations = [ts.shape[0] for ts in timeseries]
        self.duration_min = min(self.durations)

        self.timeseries = timeseries
        self.num_batches = self.duration_min // self.batch_duration
        self.idx_starts_raw = [
            torch.arange(0, d - self.batch_duration, self.batch_duration) 
            for d in self.durations
        ]

        self.idx_datasets_ = torch.arange(self.num_datasets)

        self.current_batch_ = 0
        self.idx_starts_ = None

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.current_batch_ = 0
        
        if self.shuffle_startIdx:
            ## Roll (modulo % the duration - batch_duration) and randperm each idx_starts_raw
            rolls = [int(torch.randint(0, d, (1,))[0]) for d in self.durations]
            self.idx_starts_ = np.array([
                (torch.roll(self.idx_starts_raw[idx], rolls[idx]) % (self.durations[idx] - self.batch_duration))[torch.randperm(len(self.idx_starts_raw[idx]))]
                for idx in torch.arange(self.num_datasets)
            ], dtype=object)

            self.current_startIdx_ = torch.zeros(len(self.idx_starts_), dtype=torch.long)
        else:
            self.idx_starts_ = self.idx_starts_raw
            self.current_startIdx_ = torch.zeros(len(self.idx_starts_), dtype=torch.long)

        return self

    def _next_startIdx(self, idx_dataset):
        if self.current_startIdx_[idx_dataset].item() >= len(self.idx_starts_[idx_dataset]):
            raise StopIteration
        
        idx_start = self.idx_starts_[idx_dataset][self.current_startIdx_[idx_dataset]]
        self.current_startIdx_[idx_dataset] += 1
        return idx_start

    def __next__(self):
        if self.current_batch_ >= self.num_batches:
            raise StopIteration

        # Shuffle datasets and starting indices if required
        if self.shuffle_datasets:
            self.idx_datasets_ = torch.arange(self.num_datasets)[torch.randint(0, self.num_datasets, (self.batch_size,))]

        # Prepare batch data
        batch_data = [
            self.timeseries[idx_data][(w := self._next_startIdx(idx_data)):w + self.batch_duration,:]
            for idx_data in self.idx_datasets_
        ]

        self.current_batch_ += 1
        return torch.stack(batch_data, dim=0)
    
    def to(self, device):
        self.device = device
        self.timeseries = [ts.to(device) for ts in self.timeseries]
        return self


class BinaryClassification_RNN():
    def __init__(
        self,
        features=1,
        batch_size=50,
        batch_duration=900,
        num_epochs=1000,
        hidden_size=20,
        num_layers=2,
        dropout=0.0,
        architecture='RNN',
        val_check_period=10,
        X_val=None,
        y_val=None,
        device='cpu',
        lr=0.01,
        verbose=True,
    ):
        self.batch_size = batch_size
        self.batch_duration = batch_duration
        self.num_epochs = num_epochs

        self.X_val = X_val
        self.y_val = y_val

        self.device = device
        self.val_check_period=val_check_period
        self.verbose = verbose

        if (self.X_val is not None) and (self.y_val is not None):
            assert self.X_val.ndim in [2], f'X_val must be 2D. Got {self.X_val.ndim}D'
            assert self.y_val.ndim in [1], f'y_val must be 1D. Got {self.y_val.ndim}D'
            assert self.X_val.shape[0] == self.y_val.shape[0], f'X_val and y_val must have the same number of samples. Got {self.X_val.shape[0]} and {self.y_val.shape[0]}'
            self.X_val = torch.as_tensor(self.X_val, dtype=torch.float32, device=self.device)[None,:,:]
            self.y_val = torch.as_tensor(self.y_val, dtype=torch.float32, device=self.device)[None,:,None]
            self.run_validation = True
        else:
            self.run_validation = False
    
        self.model = RegressionRNN(
            features=features, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            output_size=1,
            batch_first=True,
            architecture=architecture,
            kwargs_architecture={},
            nonlinearity='tanh',
            bias=True,
            dropout=dropout,
            bidirectional=False,
            device=self.device,
            dtype=torch.float32,
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_all = {}
        self.loss_val_all = {}
        self.epoch = 0

    def fit(
        self,
        X,
        y,
    ):
        if isinstance(X, list):
            assert all([x.ndim == 2 for x in X]), f'All X must be 2D. Got {[x.ndim for x in X]}'
            assert all([x.shape[1] == X[0].shape[1] for x in X]), f'All X must have the same number of features. Got {[x.shape for x in X]}'
            assert all([y.ndim == 1 for y in y]), f'All y must be 1D. Got {[y.ndim for y in y]}'
            timeseries = [torch.as_tensor(np.concatenate((y[:,None], x), axis=1), dtype=torch.float32, device=self.device) for x, y in zip(X, y)]
        else:
            assert isinstance(X, (np.ndarray, torch.Tensor)), f'X must be a list of 2D arrays or a 2D array. Got {type(X)}'
            assert X.ndim == 2, f'X must be 2D. Got {X.ndim}'
            assert X.shape[0] == y.shape[0], f'X and y must have the same number of samples. Got {X.shape[0]} and {y.shape[0]}'
            assert y.ndim == 1, f'y must be 1D. Got {y.ndim}'
            timeseries = [torch.as_tensor(np.concatenate((y[:,None], X), axis=1), dtype=torch.float32, device=self.device)]

        self.dataloader_XYcat = Dataloader_MultiTimeSeries(
            timeseries=timeseries,
            batch_size=self.batch_size,
            batch_duration=self.batch_duration,
            shuffle_datasets=True,
            shuffle_startIdx=True,
        )            
        
        num_epochs = self.epoch + self.num_epochs
        for self.epoch in tqdm(range(self.epoch, num_epochs), disable=not self.verbose):
            self.model.train()
            for iter_x, batch in enumerate(self.dataloader_XYcat):
                x_batch = batch[:,:,1:].to(self.device)
                y_batch = batch[:,:,0:1].to(self.device)
            
                self.optimizer.zero_grad()
                outputs, hidden = self.model.forward(
                    x=x_batch,
                    hidden_initialization='orthogonal',
                ) 
                # Ensure output dimensions match target dimensions
                loss = self.criterion(outputs.view(-1, 1), y_batch.view(-1,1))
                loss.backward()
                self.optimizer.step()
            
                self.loss_all[(self.epoch, iter_x)] = loss.item()
            
            if self.epoch >= num_epochs:
                break

            if self.run_validation and ((self.epoch % self.val_check_period) == 0):
                self.model.eval()
                y_hat_val, hidden = self.model.forward(
                    x=self.X_val,
                    hidden_initialization='orthogonal',
                )
                loss_val = self.criterion(y_hat_val.view(-1, 1), self.y_val.view(-1,1)).item()
                self.loss_val_all[self.epoch] = loss_val
            else:
                loss_val = torch.nan
                
            if self.verbose:
                print(f'Epoch [{self.epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss val: {loss_val:.4f}')

    def predict_proba(self, X):
        self.model.eval()
        y_hat, hidden = self.model.forward(
            x=torch.as_tensor(X, dtype=torch.float32, device=self.device)[None,:,:],
            hidden_initialization='orthogonal',
        )
        proba = torch.sigmoid(y_hat).detach().cpu().numpy()[0,:,0]
        proba_onehot = np.stack([1-proba, proba], axis=-1)
        return proba_onehot
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=-1)
    
    def to(self, device):
        self.device = device
        self.model.to(device)
        
        ## Move optimizer parameters to new device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        ## Move X_val and y_val to new device
        if hasattr(self, 'X_val'):
            if isinstance(self.X_val, torch.Tensor):
                self.X_val = self.X_val.to(device)
        if hasattr(self, 'y_val'):
            if isinstance(self.y_val, torch.Tensor):
                self.y_val = self.y_val.to(device)

        ## Move dataloader to new device
        if hasattr(self, 'dataloader_XYcat'):
            self.dataloader_XYcat.to(device)
            
        return self
    

class MLPRegressor_sk_tunable(sklearn.neural_network.MLPRegressor):
    """
    Subclass of sklearn.neural_network.MLPRegressor with individually tunable layer sizes.
    Name each layer: 'n_units_layer_1', 'n_units_layer_2', 'n_units_layer_n' where n is the layer number.
    """
    def __init__(self, **kwargs):
        # Extract 'n_units_layer_i' arguments
        n_units_layers = {}
        other_kwargs = {}
        for key, value in kwargs.items():
            if key.startswith('n_units_layer_'):
                try:
                    layer_num = int(key[len('n_units_layer_'):])
                    n_units_layers[layer_num] = int(round(float(value)))
                except ValueError:
                    raise ValueError(f"Invalid layer specification: {key}")
            else:
                other_kwargs[key] = value

        if not n_units_layers:
            raise ValueError("At least one 'n_units_layer_i' argument is required.")

        layer_numbers = sorted(n_units_layers.keys())

        # Construct hidden_layer_sizes tuple
        hidden_layer_sizes = tuple(n_units_layers[i] for i in layer_numbers if n_units_layers[i] > 0)

        # Initialize the superclass with the constructed hidden_layer_sizes
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, **other_kwargs)

    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import numpy as np
# from tqdm import tqdm
import os


class MLPRegressor(nn.Module):
    """
    A flexible and efficient MLP regressor with various modern training tricks.
    """
    def __init__(self, input_dim, output_dim=1, hidden_dims=[128, 64, 32],
                 activation='relu', dropout=0.0,
                 batch_norm=False, residual=False, init_type='kaiming', **kwargs):
        super(MLPRegressor, self).__init__()

        self.activation = self._get_activation(activation)
        self.init_type = init_type
        self.residual = residual
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList() if dropout > 0 else None
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                self.dropouts.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)
        # Initialize weights
        self.apply(self._init_weights)

    def _get_activation(self, activation):
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.01, inplace=True),
            'elu': nn.ELU(inplace=True),
            'selu': nn.SELU(inplace=True),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'none': nn.Identity()
        }
        return activations.get(activation.lower(), nn.ReLU(inplace=True))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif self.init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif self.init_type == 'normal':
                nn.init.normal_(m.weight, 0, 0.02)
            elif self.init_type == 'uniform':
                nn.init.uniform_(m.weight, -0.1, 0.1)
            elif self.init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = x
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms:
                x = self.batch_norms[idx](x)
            x = self.activation(x)
            if self.residual and x.shape == residual.shape:
                x = x + residual
            if self.dropouts:
                x = self.dropouts[idx](x)
            residual = x
        x = self.output_layer(x)
        return x


def train_model(model, train_loader, val_loader=None,
                criterion=nn.MSELoss(), optimizer_name='adam', lr=1e-3,
                L1_penalty=0.0, epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu',
                gradient_clip=None, verbose=True, L2_penalty=0.0,
                validation_period=1,
                use_swa=False, swa_start=10, swa_freq=5, swa_lr=1e-3):
    """
    Train the model with various modern training techniques, including SWA.
    """
    # Move model to device
    model = model.to(device)

    # Choose optimizer
    optimizers = {
        'sgd': SGD(model.parameters(), lr=lr, weight_decay=L2_penalty),
        'adam': Adam(model.parameters(), lr=lr, weight_decay=L2_penalty),
        'adamw': AdamW(model.parameters(), lr=lr, weight_decay=L2_penalty),
    }
    optimizer = optimizers.get(optimizer_name.lower(), Adam(model.parameters(), lr=lr, weight_decay=L2_penalty))

    # Initialize losses dictionary
    losses = {'train': {}, 'val': {}}

    # SWA setup
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    else:
        swa_model = None

    ## Make tqdm bar
    if verbose:
        pbar = tqdm(range(epochs), desc='Training', unit='epochs')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if L1_penalty > 0:
                loss += L1_penalty * sum(p.abs().sum() for p in model.parameters())

            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses)
        losses['train'][epoch] = avg_train_loss

        # Validation step
        if val_loader and epoch % validation_period == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            losses['val'][epoch] = avg_val_loss

        # SWA step
        if use_swa:
            swa_scheduler.step()
            if epoch >= swa_start and (epoch - swa_start) % swa_freq == 0:
                swa_model.update_parameters(model)

        ## Update tqdm bar
        if verbose:
            ## Print losses as well
            pbar.set_postfix({
                'train_loss': list(losses['train'].values())[-1],
                'val_loss': list(losses['val'].values())[-1] if len(losses['val']) > 0 else np.nan,
            })
            pbar.update(1)

    # Update BN for SWA model
    if use_swa:
        update_bn(train_loader, swa_model, device=device)
        model = swa_model

    return model, losses