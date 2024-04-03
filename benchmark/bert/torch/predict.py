import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import benchmark
from benchmark import torch_utils

torch.set_float32_matmul_precision('high')

def run(batch_size=benchmark.BERT_BATCH_SIZE):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=2,
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()  # Set model to evaluation mode

    # Assuming torch_utils.get_train_dataset_for_text_classification returns a PyTorch Dataset of texts
    dataset = torch_utils.get_train_dataset_for_text_classification(
        tokenizer,
        batch_size=batch_size,
        seq_len=benchmark.BERT_SEQ_LENGTH,
    )

    print(dataset[0].keys)

    # Assuming dataset is a PyTorch Dataset with a __getitem__ method that returns a single text string
    texts = dataset[0]  # Extract texts assuming dataset[i] returns (text, label)
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=benchmark.BERT_SEQ_LENGTH, return_tensors="pt")
    inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

    # Warm-up inference
    with torch.no_grad():
        model(**{key: val[:batch_size] for key, val in inputs.items()})

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for step in range(benchmark.NUM_STEPS):
            model(**inputs)
    end_time = time.time()

    total_time = end_time - start_time
    inferencing_per_step = total_time / benchmark.NUM_STEPS * 1000

    return inferencing_per_step

if __name__ == "__main__":
    benchmark.benchmark(run)
