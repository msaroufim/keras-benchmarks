import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import StaticCache
torch.set_default_device("cuda")
torch.set_float32_matmul_precision('high')
import benchmark
from benchmark import torch_utils

import torch
def no_op(*args, **kwargs):
     pass

torch._C._set_storage_access_error_msg = no_op

def run(batch_size=benchmark.GEMMA_BATCH_SIZE):
    preset = "google/gemma-7b"
    model = AutoModelForCausalLM.from_pretrained(
        preset, torch_dtype=torch_utils.get_torch_dtype(benchmark.FLOAT_A100)
    ).cuda()
    # model._setup_cache(StaticCache, batch_size, max_cache_len=benchmark.GEMMA_MAX_LENGTH)

    model.forward = torch.compile(model.forward, mode=torch_utils.COMPILE_MODE)
    tokenizer = AutoTokenizer.from_pretrained(preset)
    tokenizer.pad_token = tokenizer.eos_token

    return torch_utils.generate(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=benchmark.GEMMA_MAX_LENGTH,
    )


if __name__ == "__main__":
    benchmark.benchmark(run)
