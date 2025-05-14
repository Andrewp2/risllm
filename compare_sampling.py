import time, random, torch, re, string, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from reservoir_sampler import ReservoirLogitsProcessor
from tqdm import tqdm

# Define model_name globally
model_name = "qwen/Qwen3-8B"

def evaluate_mmlu(model, tokenizer, examples, use_reservoir, num_samples=100, max_new_tokens=150, K=64, R=32, temperature=1.0, final_answer_temperature_override=None):
    """Evaluate model on MMLU benchmark using either standard sampling or reservoir sampling, with Chain-of-Thought prompting using a two-step generation process."""
    proc = ReservoirLogitsProcessor(K=K, R=R, device=model.device) if use_reservoir else None

    # Select a subset of examples for evaluation
    if len(examples) > num_samples:
        random.seed(42)  # For reproducibility
        examples = random.sample(examples, num_samples)

    correct = 0
    total = 0

    for i_example, example in enumerate(tqdm(examples, leave=False)):
        # Format the question as a prompt
        question = example['question']
        choices = [example['choices'][i] for i in range(len(example['choices']))]

        # Create a prompt with the question and choices, encouraging Chain-of-Thought
        prompt_reasoning = f"Question: {question}\n\nChoices:\n"
        for i, choice in enumerate(choices):
            prompt_reasoning += f"{string.ascii_uppercase[i]}. {choice}\n"
        prompt_reasoning += "\nPlease think step-by-step to arrive at the correct answer.\n\nReasoning: "

        # Tokenize the reasoning prompt
        inputs_reasoning = tokenizer(prompt_reasoning, return_tensors="pt", add_special_tokens=True)
        input_ids_reasoning = inputs_reasoning.input_ids.to(model.device)
        attention_mask_reasoning = inputs_reasoning.attention_mask.to(model.device)

        # Generate reasoning
        max_reasoning_tokens = max_new_tokens - 20 # Reserve 20 tokens for the final answer part
        with torch.no_grad():
            outputs_reasoning = model.generate(
                input_ids_reasoning,
                attention_mask=attention_mask_reasoning,
                max_new_tokens=max_reasoning_tokens,
                logits_processor=[proc] if proc else None,
                do_sample=True,
                top_p=1.0,
                temperature=temperature,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_reasoning_text = tokenizer.decode(outputs_reasoning[0][input_ids_reasoning.shape[1]:], skip_special_tokens=True).strip()

        # Second, prompt for the final answer based on the reasoning
        prompt_final_answer = f"{prompt_reasoning}{generated_reasoning_text}\n\nBased on the reasoning above, what is the Final Answer? (Format: Final Answer: [CHOSEN_LETTER])"

        inputs_final_answer = tokenizer(prompt_final_answer, return_tensors="pt", add_special_tokens=True)
        input_ids_final_answer = inputs_final_answer.input_ids.to(model.device)
        attention_mask_final_answer = inputs_final_answer.attention_mask.to(model.device)

        # Ensure we don't exceed model's max sequence length
        # This is a simple truncation, more sophisticated handling might be needed for very long reasoning
        max_model_len = getattr(model.config, 'max_position_embeddings', 2048)
        if input_ids_final_answer.shape[1] > max_model_len:
            input_ids_final_answer = input_ids_final_answer[:, -max_model_len:]
            attention_mask_final_answer = attention_mask_final_answer[:, -max_model_len:]

        # Determine temperature for the final answer generation step
        final_answer_temp = temperature # Default to reasoning temperature
        if final_answer_temperature_override is not None:
            final_answer_temp = final_answer_temperature_override

        with torch.no_grad():
            outputs_final_answer = model.generate(
                input_ids_final_answer,
                attention_mask=attention_mask_final_answer,
                max_new_tokens=20, # Sufficient for "Final Answer: X"
                logits_processor=[proc] if proc else None, # Apply same processor if active
                do_sample=True,
                top_p=1.0, 
                temperature=final_answer_temp, # Use determined final answer temperature
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode the generated final answer text
        generated_final_answer_text = tokenizer.decode(outputs_final_answer[0][input_ids_final_answer.shape[1]:], skip_special_tokens=True).strip()

        # --- DIAGNOSTIC PRINT ---
        if i_example < 3:
            print(f"\n--- Example {i_example+1} ({'Reservoir' if use_reservoir else 'Top-p'}) ---")
            print(f"Reasoning Prompt:\n{prompt_reasoning}")
            print(f"Generated Reasoning Text:\n{generated_reasoning_text}")
            print(f"Final Answer Prompt (condensed input):\nBased on the reasoning above, what is the Final Answer? (Format: Final Answer: [CHOSEN_LETTER])")
            print(f"Generated Final Answer Text:\n{generated_final_answer_text}")
            print("---------------------------")
        # --- END DIAGNOSTIC PRINT ---

        # Extract the answer (look for 'Final Answer: X' in the second generation's output)
        answer_match = re.search(r'Final Answer:\s*([A-Z])', generated_final_answer_text, re.IGNORECASE)
        if answer_match:
            predicted_answer = answer_match.group(1).upper()
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Load MMLU dataset (using a subset for faster evaluation)
    mmlu_dataset = load_dataset("cais/mmlu", "all")
    num_eval_examples = 100 # Number of examples to evaluate from MMLU 'test' split
    examples = list(mmlu_dataset['test'].shuffle(seed=42).select(range(num_eval_examples)))

    # Run evaluations
    print(f"Evaluating on {num_eval_examples} MMLU examples using {model_name}...")
    
    start_time = time.time()
    # For top-p: Reasoning Temp = 1.0, Final Answer Temp = 0.1
    acc_base = evaluate_mmlu(model, tokenizer, examples, use_reservoir=False, num_samples=num_eval_examples, temperature=1.0, max_new_tokens=150, final_answer_temperature_override=0.1)
    time_base = time.time() - start_time

    start_time = time.time()
    # For reservoir: Reasoning Temp = 1.0, Final Answer Temp = 0.5
    acc_res = evaluate_mmlu(model, tokenizer, examples, use_reservoir=True, num_samples=num_eval_examples, K=64, R=32, temperature=1.0, max_new_tokens=150, final_answer_temperature_override=0.5)
    time_res = time.time() - start_time

    # Print results
    print(f"top-p sampling  accuracy: {acc_base:.4f}   ({time_base:.1f}s)")
    print(f"reservoir       accuracy: {acc_res:.4f}   ({time_res:.1f}s)")
    if acc_base > 0:
        print(f"reservoir / baseline = {acc_res/acc_base:.3f}")

if __name__ == "__main__":
    main()
