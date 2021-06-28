import torch
import torchvision

# from util_test import helper_setUp_CIFAR10_same
from pytorchfi.core import fault_injection
import timm

if __name__ == "__main__":
    BATCH_SIZE = 4
    WORKERS = 1
    IMG_SIZE = 224
    USE_GPU = False
    # model, dataset = helper_setUp_CIFAR10_same(BATCH_SIZE, WORKERS)
    # dataiter = iter(dataset)
    torch.manual_seed(5)
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.eval()
    images = torch.rand((BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE))
    torch.no_grad()
    output = model(images)
    gold_label = list(torch.argmax(output, dim=1))[0].item()
    print("Golden Label:", gold_label)
    # p = fault_injection(model, IMG_SIZE, IMG_SIZE, BATCH_SIZE, use_cuda=USE_GPU)
    p = fault_injection(
        model,
        # IMG_SIZE,
        BATCH_SIZE,
        [3, 224, 224],
        layer_types=[torch.nn.Conv2d, torch.nn.Linear],
        use_cuda=False,
    )

    (b, layer, C, H, W, err_val) = ([0], [2], [190], [3000], [None], [10000])
    inj = p.declare_neuron_fi(batch=b, layer_num=layer, c=C, h=H, w=W, value=err_val)
    inj_output = inj(images)
    # print(p.print_pytorchfi_layer_summary())
