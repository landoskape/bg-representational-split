# library imports
from collections import OrderedDict
import torch

# import rastermap
from cornet import get_model


def get_cornet_model(*args, output_dim=1000, **kwargs):
    """wrapper of get_model from cornet to return the model rather than a DataParallel object"""
    out_shape = kwargs.pop("out_shape", None)
    model = get_model(*args, **kwargs)
    model = model.module
    if output_dim != 1000:
        update_output_dim(model, output_dim=output_dim)
    if args[0] == "RT" and out_shape is not None:
        model.V1.out_shape = out_shape[0]
        model.V2.out_shape = out_shape[1]
        model.V4.out_shape = out_shape[2]
        model.IT.out_shape = out_shape[3]
    return model


def update_output_dim(model, output_dim=2):
    """update the number of output dimensions of the model (in the decoder layer)"""
    model.decoder.linear = torch.nn.Linear(512, output_dim)


def set_gradient(model, requires_grad=["decoder"]):
    for name, param in model.named_parameters():
        if any([rg in name for rg in requires_grad]):
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_hidden(model, input_tensor, flatten=True):
    """method to get hidden activations of CORnet model"""
    activations = OrderedDict()

    def hook(module, input, output, name):
        activations[name] = output

    # Register hooks on each layer
    for name, module in model.named_children():
        module.register_forward_hook(lambda module, input, output, name=name: hook(module, input, output, name))

    # Perform forward pass to trigger hooks
    _ = model(input_tensor)

    # Get activations and put in list of tensors
    hidden = [activations[a] for a in activations]

    # Flatten if requested
    if flatten:

        def get_out_incase_recurrent(h):
            if type(h) == torch.Tensor:
                return h
            elif type(h) == tuple:
                return h[0]

        hidden = [get_out_incase_recurrent(h).flatten(start_dim=1) for h in hidden]

    return hidden


class bgnet(torch.nn.Module):
    """
    bgnet is a simple feedforward relu network with dropout designed to produce logits
    that can be used for image classification
    """

    def __init__(self, input_dim=1000, hidden_widths=[100, 50], output_dim=2, dropout=0.5, scale=None):
        super().__init__()
        self.scale = scale

        self.layers = torch.nn.ModuleList()

        # add input layer
        input_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_widths[0]),
            torch.nn.ReLU(),
        )

        self.layers.append(input_layer)

        # add hidden layers
        for h_in, h_out in zip(hidden_widths[:-1], hidden_widths[1:]):
            c_layer = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(h_in, h_out),
                torch.nn.ReLU(),
            )
            self.layers.append(c_layer)

        # initialize weight scale of hidden layers
        self.apply(self.__initweights__)

        # create output layer
        output_layer = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_widths[-1], output_dim),
        )

        # initialize appropriately
        output_layer[1].weight.data.normal_(mean=0.0, std=1 / torch.sqrt(torch.tensor(hidden_widths[-1])))
        output_layer[1].bias.data.zero_()

        # add output layer
        self.layers.append(output_layer)

    def __initweights__(self, m):
        """simple method for initizalizing weight scale of linear layers"""
        with torch.no_grad():
            if type(m) == torch.nn.Linear:
                m.bias.data.zero_()
                if self.scale is not None:
                    m.weight.data.normal_(mean=0.0, std=self.scale)

    def forward(self, x, store_hidden=False):
        """standard forward pass of all layers with option of storing hidden activations (and output)"""
        self.hidden = []  # always reset so as to not keep a previous forward pass accidentally
        x = x.flatten(start_dim=1)  # always flatten image dimensions
        for layer in self.layers:
            x = layer(x)  # pass through next layer
            if store_hidden:
                self.hidden.append(x)
        return x
