# training/train_lora_generator.py
import os, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch

def main(base_model, train_jsonl, output_dir, epochs=1, batch_size=1, qlora=False):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    ds = load_dataset("json", data_files={"train": train_jsonl})["train"]

    def tokenize_fn(ex):
        # ex["text"] is prompt + target
        out = tokenizer(ex["text"], truncation=True, max_length=1024)
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        load_in_4bit=qlora,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    if qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch"
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=data_collator)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Saved LoRA model to", output_dir)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--train_jsonl", default="data/train_lora.jsonl")
    p.add_argument("--output_dir", default="models/lora_generator")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--qlora", action="store_true")
    args=p.parse_args()
    main(args.base_model, args.train_jsonl, args.output_dir, args.epochs, args.batch_size, args.qlora)
