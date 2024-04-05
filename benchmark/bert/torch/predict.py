import time

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

import benchmark
from benchmark import torch_utils
import torch
torch.set_float32_matmul_precision('high')
MODE="reduce-overhead"
## Additional inductor flags
import torch._inductor.config

# helps autotuning
# torch._inductor.config.coordinate_descent_tuning = True

# speeds up warm compile times, no impact on benchmark
torch._inductor.config.fx_graph_cache = True

def run_inference(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs

def run(batch_size=benchmark.BERT_BATCH_SIZE):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = torch_utils.get_train_dataset_for_text_classification(
        tokenizer,
        batch_size=batch_size,
        seq_len=benchmark.BERT_SEQ_LENGTH,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=2,
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.compile(model, mode=MODE)

    # Select a batch from the dataset
    batch = dataset.select(list(range(batch_size)))
    inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=benchmark.BERT_SEQ_LENGTH)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Warmup
    run_inference(model, inputs)

    # Run inference
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    for _ in range(benchmark.NUM_STEPS + 1):
        run_inference(model, inputs)
    end_time.record()
    torch.cuda.synchronize()
    return (start_time.elapsed_time(end_time)) / benchmark.NUM_STEPS

if __name__ == "__main__":
    benchmark.benchmark(run)
