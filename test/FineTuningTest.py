from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import json

# 1. Cargar dataset desde tu archivo JSONL
def load_jsonl_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return Dataset.from_list(data)

dataset = load_jsonl_dataset("/mnt/data/dataset.jsonl")

# 2. Formato de entrada tipo prompt
def format_example(example):
    return {
        "text": f"<|user|>\n{example['instruction']}\n{example.get('input', '')}\n<|assistant|>\n{example['output']}"
    }

formatted_dataset = dataset.map(format_example)

# 3. Tokenización
model_id = "deepseek-ai/deepseek-llm-8b-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # importante para evitar errores

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = formatted_dataset.map(tokenize)

# 4. Cargar modelo y aplicar LoRA
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # comunes en modelos transformer
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# 5. Entrenamiento
training_args = TrainingArguments(
    output_dir="./deepseek-8b-lora",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    fp16=True,
    learning_rate=2e-4,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# 6. Guardar modelo LoRA adaptado
model.save_pretrained("deepseek-8b-lora")
tokenizer.save_pretrained("deepseek-8b-lora")
