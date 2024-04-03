import segment_anything_fast as segment_anything
import torch

import benchmark
from benchmark import torch_utils
torch.set_float32_matmul_precision('high')
torch.set_default_dtype(torch.bfloat16)
MODE="max-autotune"
## Additional inductor flags
import torch._inductor.config

# helps autotuning
torch._inductor.config.coordinate_descent_tuning = True

# speeds up warm compile times, no impact on benchmark
torch._inductor.config.fx_graph_cache = True

HUGE_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)
HUGE_BUILD = segment_anything.build_sam_fast_vit_h
HUGE_LOCAL = "/tmp/sam_h.pth"
LARGE_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
)
LARGE_BUILD = segment_anything.build_sam_fast_vit_l
LARGE_LOCAL = "/tmp/sam_l.pth"
BASE_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)
BASE_BUILD = segment_anything.build_sam_fast_vit_b
BASE_LOCAL = "/tmp/sam_b.pth"

URL = HUGE_URL
LOCAL = HUGE_LOCAL
build_sam = HUGE_BUILD


def get_dataset(batch_size):
    input_image = torch.Tensor(batch_size, 3, 1024, 1024).cuda()
    y_true = torch.Tensor(batch_size, 256, 64, 64).cuda()
    return input_image, y_true


def train(model, input_image, y_true):
    optimizer = torch.optim.Adam(model.parameters())

    def train_fn(model, input_image, y_true):
        optimizer.zero_grad()
        y_pred = model(input_image)
        loss = torch.nn.MSELoss()(y_pred, y_true)
        loss.backward()
        optimizer.step()

    compiled_train_fn = torch.compile(train_fn, mode=MODE)

    compiled_train_fn(model, input_image, y_true)

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    for _ in range(benchmark.NUM_STEPS):
        compiled_train_fn(model, input_image, y_true)
    end_time.record()

    # CUDA is async so wait for everything to finish running
    torch.cuda.synchronize()

    # cuda.Event returns times in ms already
    return (start_time.elapsed_time(end_time)) / benchmark.NUM_STEPS


def run(batch_size=benchmark.SAM_FIT_BATCH_SIZE):
    benchmark.download_file(URL, LOCAL)
    model = build_sam(checkpoint=LOCAL).cuda()
    input_image, y_true = get_dataset(batch_size)

    return train(model.image_encoder, input_image, y_true)


if __name__ == "__main__":
    benchmark.benchmark(run)
