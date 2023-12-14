import torch

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