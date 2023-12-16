import copy

import torch
import numpy as np

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
        input_size=1, 
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
        self.input_size = input_size
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
