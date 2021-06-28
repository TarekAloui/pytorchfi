import torch
import torchvision.models as models
from pytorchfi.core import fault_injection
import timm
import pytest


class TestLayers:
    """
    Testing PyTorchFI.Core example client.
    """

    def setup_class(self):
        torch.manual_seed(5)

        self.H = 224
        self.W = 224
        self.BATCH_SIZE = 4

        self.IMAGE = torch.rand((self.BATCH_SIZE, 3, self.H, self.W))

        self.USE_GPU = False

        self.softmax = torch.nn.Softmax(dim=1)

        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.model.eval()

        # Error free inference to gather golden value
        self.output = self.model(self.IMAGE)
        self.golden_softmax = self.softmax(self.output)
        self.golden_label = list(torch.argmax(self.golden_softmax, dim=1))[0].item()

    # def test_golden_inference(self):
    #     print("------------------------------------")
    #     print("Golden", self.golden_label)
    #     print("------------------------------------")
    #     assert self.golden_label == 556

    def test_single_conv_neuron(self):

        p = fault_injection(
            self.model,
            self.BATCH_SIZE,
            input_shape=[3, self.H, self.W],
            layer_types=[torch.nn.Linear],
            use_cuda=self.USE_GPU,
        )

        (b, layer, C, H, W, err_val) = ([0], [2], [None], [190], [3000], [10000])
        inj = p.declare_neuron_fi(
            batch=b, layer_num=layer, c=C, h=H, w=W, value=err_val
        )
        inj_output = inj(self.IMAGE)
        inj_softmax = self.softmax(inj_output)
        inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()

        print("Inj Label", inj_label)
        print("------------------------------------")

        # assert inj_label == 120

    def test_single_linear_layer(self):
        p = fault_injection(
            self.model,
            self.BATCH_SIZE,
            input_shape=[3, self.H, self.W],
            layer_types=[torch.nn.Linear],
            use_cuda=self.USE_GPU,
        )

        print("TOTAL LAYERS", p.get_total_layers())
        print("LAYER DIM", p.get_layer_dim(2))
        print("LAYER TYPE", p.get_layer_type(2))
        print("Summary", p.print_pytorchfi_layer_summary())
        print("------------------------------------")

        # assert p.get_total_layers() == 3
        # assert p.get_layer_dim(2) == 2
        # assert p.get_layer_type(2) == torch.nn.Linear
