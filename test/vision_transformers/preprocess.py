###
# This file is used to execute a profiling pass

import timm
import torch
import torch.tensor as tensor
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import os.path
import pickle
import argparse

IMAGENET_PATH = "/data/imagenet/"
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256

activations = []


def save_activations(module, input, output):
    activations.append(output)


def gather_min_max_per_layer(
    model,
    data_iter,
    batch_size,
    precision="FP16",
    debug=False,
    verbose=False,
):
    global activations
    layer_max = torch.Tensor([]).cuda().half()  # ADDED
    layer_min = torch.Tensor([]).cuda().half()  # ADDED
    fmap_ranges = []  # TODO: split up by layer

    # register forward hook to the model
    handles = []
    for param in model.modules():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
            handles.append(param.register_forward_hook(save_activations))

    # main loops to gather ranges
    processed_elements = 0
    batch_num = 0
    total_fmaps = 0

    print("NUMBER INPUTS", len(data_iter))

    for input_data in tqdm(data_iter):

        # prepare the next batch for inference
        images, labels = input_data
        input_element = images.cuda()
        if precision == "FP16":
            input_element = input_element.half()

        activations = []  # reset before every inference
        model(input_element)  # run an inference

        # Range gathering: iterate through each layer

        min_vals = (
            torch.Tensor(list(map(lambda layer: layer.min().item(), activations)))
            .cuda()
            .half()
        )
        max_vals = (
            torch.Tensor(list(map(lambda layer: layer.max().item(), activations)))
            .cuda()
            .half()
        )
        if batch_num == 0:
            layer_max = max_vals
            layer_min = min_vals
        else:
            layer_max = torch.max(layer_max, max_vals)
            layer_min = torch.min(layer_min, min_vals)

        # for layer in range(len(activations)):
        #     # TODO find better way to find max and min in 4D tensor (activations[layer])
        #     max_val = activations[layer].max()
        #     min_val = activations[layer].min()

        #     # compare max_val and min_val per fmap to choose the abs max
        #     actual_max = []
        #     for i in range(len(min_val[0])):
        #         min_val_fmap = abs(min_val[0][i].item())
        #         max_val_fmap = abs(max_val[0][i].item())

        #         if max_val_fmap > min_val_fmap:
        #             actual_max.append(max_val_fmap)
        #         else:
        #             actual_max.append(min_val_fmap)

        #     if batch_num == 0:
        #         fmap_ranges.append(actual_max)
        #     else:
        #         for i, prev_max in enumerate(fmap_ranges[layer]):
        #             new_max = actual_max[i]
        #             if prev_max >= new_max:
        #                 continue
        #             else:
        #                 fmap_ranges[layer][i] = new_max
        processed_elements += len(labels)
        batch_num += 1
        torch.cuda.empty_cache()

    # remove hooks
    for i in range(len(handles)):
        handles[i].remove()
    del activations

    actual_max = torch.max(torch.abs(layer_min), torch.abs(layer_max))
    return layer_min, layer_max, actual_max


def load_dataiter(batch_sz):
    traindir = os.path.join(IMAGENET_PATH, "train")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=batch_sz,
        shuffle=False,
        num_workers=8,  # Tells pytorch the number of cores to use
    )
    dataiter = iter(val_loader)
    return dataiter


# finds the max value for each layer
def get_layer_ranges(range_data):
    layer_max = {}
    for layer in range(len(range_data)):
        layer_max[layer] = -1
        for fmap in range(len(range_data[layer])):
            curr_val = range_data[layer][fmap]
            if layer_max[layer] < curr_val:
                layer_max[layer] = curr_val
    return layer_max


def save_data(path, file_name, data):
    if not os.path.exists(path):
        os.makedirs(path)
    output = path + file_name + ".p"
    f = open(output, "wb")
    pickle.dump(data, f)
    f.close()


if __name__ == "__main__":

    # torch.cuda.set_device(0)

    # Parsing Arguments
    parser = argparse.ArgumentParser(
        description="Executing Profiling pass on a timm supported model"
    )

    parser.add_argument("model", help="The model to be used for profiling")
    parser.add_argument(
        "-b", "--batchsz", dest="batchsz", help="The batch size", type=int
    )
    args = parser.parse_args()
    model_name = args.model if args.model else "vit_base_patch16_224"
    batchsz = args.batchsz if args.batchsz else BATCH_SIZE

    # Loading model and dataset
    model = timm.create_model(model_name, pretrained=True).cuda().half()
    dataiter = load_dataiter(batchsz)

    # Executing profiling pass
    layer_min, layer_max, actual_max = gather_min_max_per_layer(
        model, dataiter, batchsz
    )
    ranges = actual_max.cpu().numpy().tolist()

    path = "/n/home09/taloui/scratch/pytorchfi/test/vision_transformers/profile/"
    save_data(path, model_name + "_range_data_layer", ranges)

    # # CSV

    # f = open(path + "range_data" + ".csv", "w+")
    # for i in range(len(range_data)):
    #     for j in range(len(range_data[i])):
    #         outputString = "%d, %d, %f\n" % (i, j, range_data[i][j])
    #         f.write(outputString)
    # f.close()

    f = open(path + model_name + "_range_data" + "_layer.csv", "w+")
    for i in range(len(ranges)):
        outputString = "%d, %f\n" % (i, ranges[i])
        f.write(outputString)
    f.close()
