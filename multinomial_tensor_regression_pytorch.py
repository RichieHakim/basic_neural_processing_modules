import torch
from torch import nn
from torch.autograd import Variable

from torch.optim import Adam, LBFGS
from torch.utils.data import Dataset, DataLoader

import tensorly as tl

def CP_logistic_regression(X, 
                            y, 
                            rank=5,
                            weights=None,
                            lambda_L2=0.01, 
                            max_iter=100, 
                            tol=1e-5, 
                            patience=10,
                            verbose=False,
                            running_loss_logging_interval=10, 
                            device='cpu',
                            **LBFGS_kwargs):

    tl.set_backend('pytorch')

    def make_BcpInit(B_dims, rank, device='cpu'):
        Bcp_init = list([torch.rand((B_dims[0], rank), requires_grad=True, device=device)])
        for ii in range(1,len(B_dims)):
            Bcp_init.append(torch.rand((B_dims[ii], rank), requires_grad=True, device=device))
        return Bcp_init

    # DEVICE = set_device(use_GPU=1)

    # X = et_traces.transpose(1,0,2)
    # y = np.squeeze(trial_type_list) -1

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    # weights = None
    # rank = 10
    # lambda_L2 = 0.005

    if weights is None:
        weights = torch.ones((rank), device=device)

    n_classes = y_oneHot.shape[1]
    B_dims = np.concatenate((np.array(X.shape[1:]), [n_classes]))
    Bcp = make_BcpInit(B_dims, rank, device)


    optimizer = torch.optim.LBFGS(Bcp,
                                lr=1, 
                                max_iter=10, 
                                max_eval=20, 
                                tolerance_grad=1e-07, 
                                tolerance_change=1e-09, 
                                history_size=100, 
                                line_search_fn="strong_wolfe"
                                )

    loss_fn = torch.nn.CrossEntropyLoss()

    def model(X, Bcp):
    #     return torch.nn.functional.softmax(torch.einsum('ijk,jr,kr,mr -> im', X, Bcp[0], Bcp[1], Bcp[2]), dim=1)
        return torch.nn.functional.softmax(tensorly.tenalg.inner(X, tensorly.cp_tensor.cp_to_tensor((weights, Bcp)), n_modes=len(Bcp)-1), dim=1)

    def L2_penalty(B_cp):
        ii=0
        for comp in B_cp:
            ii+= torch.sqrt(torch.sum(comp**2))
        return ii


    def closure():
        optimizer.zero_grad()
        y_hat = model(X, Bcp)
        loss = loss_fn(y_hat, y) + lambda_L2 * L2_penalty(Bcp)
        loss.backward()
        return loss

    loss_running = []
    for ii in range(max_iter):
        if ii%running_loss_logging_interval == 0:
            y_hat = model(X, Bcp)
            loss_running.append( loss_fn(y_hat, y).detach() )
        optimizer.step(closure)