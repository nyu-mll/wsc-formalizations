import os
import numpy


def encode_exp_name(dataset, framing, lr, bs, max_epochs, seed):
    return f"{dataset}_{framing}_lr_{lr}_bs_{bs}_max-epochs_{max_epochs}_seed_{seed}"


def decode_exp_name(exp_name):
    dataset, framing = exp_name.split("_")[:2]
    lr, bs, max_epochs, seed = exp_name.split("_")[3::2]
    lr, bs, max_epochs, seed = float(lr), int(bs), int(max_epochs), int(seed)
    return dataset, framing, lr, bs, max_epochs, seed


def make_command(dataset, framing, lr, bs, max_epochs, seed, gpu_capacity):

    exp_name = f"{encode_exp_name(dataset, framing, lr, bs, max_epochs, seed)}"
    accumulation = int(numpy.ceil(bs / gpu_capacity))
    command = (
        f"python "
        f'{os.path.join(".", "wsc-trick", "src", "main.py") }'
        f"--exp-name {exp_name} "
        f"--dataset {dataset} "
        f"--framing {framing} "
        f"--bs {bs} "
        f"--accumulation {accumulation} "
        f"--lr {lr} "
        f"--max-epochs {max_epochs} "
        f"--amp "
    )

    return command
