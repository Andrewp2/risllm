import time, random, torch, re, string, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from reservoir_sampler import ReservoirLogitsProcessor
from tqdm import tqdm

def evaluate_mmlu(model, tokenizer, examples, use_reservoir, num_samples=100, max_new_tokens=32, K=64, R=32, temperature=1.0):
    """Evaluate model on MMLU benchmark using either standard sampling or reservoir sampling."""
    proc = ReservoirLogitsProcessor(K=K, R=R, device=model.device) if use_reservoir else None

    # Select a subset of examples for evaluation
    if len(examples) > num_samples:
        random.seed(42)  # For reproducibility
        examples = random.sample(examples, num_samples)

    correct = 0
    total = 0

    for example in tqdm(examples, leave=False):
        # Format the question as a prompt
        question = example['question']
        choices = [example['choices'][i] for i in range(len(example['choices']))]

        # Create a prompt with the question and choices
        prompt = f"Question: {question}\n\nChoices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{string.ascii_uppercase[i]}. {choice}\n"
        prompt += "\nAnswer: "

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        # Generate a response
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                logits_processor=[proc] if proc else None,
                do_sample=True,
                top_p=1.0,
                temperature=temperature,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Extract the answer (look for A, B, C, D, etc.)
        answer_match = re.search(r'\b([A-Z])\b', generated_text)
        if answer_match:
            predicted_answer = answer_match.group(1)
            correct_answer = string.ascii_uppercase[example['answer']]

            if predicted_answer == correct_answer:
                correct += 1
            total += 1
        else:
            # If no clear answer found, count as incorrect
            total += 1

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    return accuracy

def main():
    # Load model and tokenizer
    model_id = "qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=True, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Load MMLU dataset (using a subset for demonstration)
    # We'll use the 5-shot dev set for evaluation
    try:
        dataset = load_dataset("cais/mmlu", "all", split="validation")
    except Exception as e:
        print(f"Error loading MMLU dataset: {e}")
        print("Falling back to a smaller subset...")
        dataset = load_dataset("cais/mmlu", "abstract_algebra", split="validation")

    # Prepare examples in the right format
    examples = []
    for item in dataset:
        example = {
            'question': item['question'],
            'choices': [item['choices'][i] for i in range(len(item['choices']))],
            'answer': item['answer']
        }
        examples.append(example)

    # Set the number of examples to evaluate
    num_eval_examples = 100  # Adjust based on your computational resources

    # Evaluate with standard top-p sampling
    t0 = time.time()
    acc_base = evaluate_mmlu(model, tokenizer, examples, use_reservoir=False, num_samples=num_eval_examples, temperature=1.0) # Standard uses defaults
    t1 = time.time()

    # Evaluate with reservoir sampling
    reservoir_K = 64
    reservoir_R = 32
    reservoir_temp = 1.0
    acc_res = evaluate_mmlu(model, tokenizer, examples, use_reservoir=True, num_samples=num_eval_examples, K=reservoir_K, R=reservoir_R, temperature=reservoir_temp)
    t2 = time.time()

    # Print results
    print(f"top-p sampling  accuracy: {acc_base:.4f}   ({t1 - t0:.1f}s)")
    print(f"reservoir       accuracy: {acc_res:.4f}   ({t2 - t1:.1f}s)")
    print(f"reservoir / baseline = {acc_res / acc_base:.3f}")

if __name__ == "__main__":
    main()
