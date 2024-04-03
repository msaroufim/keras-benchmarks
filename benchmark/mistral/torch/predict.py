import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import StaticCache

import benchmark
from benchmark import torch_utils
import torch
torch.set_float32_matmul_precision('high')
torch.set_default_device("cuda")

def run(batch_size=benchmark.MISTRAL_BATCH_SIZE):
    preset = "mistralai/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(
        preset, torch_dtype=torch_utils.get_torch_dtype(benchmark.FLOAT_A100)
    ).cuda()
    model.forward = torch.compile(model.forward, fullgraph=True, mode=torch_utils.COMPILE_MODE)

    # model = torch.compile(model, mode=torch_utils.COMPILE_MODE)
    tokenizer = AutoTokenizer.from_pretrained(preset)
    tokenizer.pad_token = tokenizer.eos_token

    return torch_utils.generate(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=benchmark.MISTRAL_MAX_LENGTH,
    )


if __name__ == "__main__":
    benchmark.benchmark(run)
