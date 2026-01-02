# ======================= HEALTHCARE CHATBOT - TRAINING CODE =======================
# This code trains the model on healthcare Q&A data using LoRA fine-tuning
# Updated to save model for production use with FastAPI backend

# ======================= 0. INSTALL =======================
# !pip install -q transformers datasets peft accelerate sentencepiece torch


# ======================= 1. MOUNT DRIVE ===================
from google.colab import drive
drive.mount('/content/drive')


# ======================= 2. CONFIG ========================
import os, json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import torch


model_dir  = "/content/drive/MyDrive/4-2 project/flan-t5-base-model-save"
data_path  = "/content/drive/MyDrive/4-2 project/healthcare_chatbot_dataset_large.csv"
output_dir = "/content/drive/MyDrive/google_flan-t5-base-trained"


os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


# ======================= 3. DOWNLOAD BASE MODEL =================
print("\n🔽 Loading base model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


tokenizer.save_pretrained(model_dir)
model.save_pretrained(model_dir)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ======================= 4. LOAD DATA ======================
print("\n📂 Loading dataset...")


df = pd.read_csv(data_path)


# 🔍 AUTO-DETECT COLUMNS
print(f"\n✅ Dataset loaded: {len(df)} rows")
print(f"📋 Available columns: {list(df.columns)}\n")


# Try to find input and output columns automatically
possible_input_cols = ['prompt', 'input', 'question', 'query', 'user_input', 'text', 'Patient', 'Question']
possible_output_cols = ['response', 'output', 'answer', 'reply', 'target', 'Doctor', 'Answer']


INPUT_COL = None
TARGET_COL = None


# Find input column
for col in possible_input_cols:
    if col in df.columns:
        INPUT_COL = col
        break


# Find output column
for col in possible_output_cols:
    if col in df.columns:
        TARGET_COL = col
        break


# If not found, manually check for your specific columns
if INPUT_COL is None and 'user_input' in df.columns:
    INPUT_COL = 'user_input'


if TARGET_COL is None and 'bot_response' in df.columns:
    TARGET_COL = 'bot_response'


# If still not found, use first two columns as fallback
if INPUT_COL is None:
    INPUT_COL = df.columns[0]
    print(f"⚠️ Using first column as input: '{INPUT_COL}'")


if TARGET_COL is None:
    TARGET_COL = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    print(f"⚠️ Using second column as output: '{TARGET_COL}'")


print(f"\n✨ Using columns:")
print(f"   Input:  '{INPUT_COL}'")
print(f"   Output: '{TARGET_COL}'")


# Show sample data
print(f"\n📄 Sample data:")
print(f"   Input:  {df[INPUT_COL].iloc[0][:100]}...")
print(f"   Output: {df[TARGET_COL].iloc[0][:100]}...\n")


# Clean data: remove NaN values
df = df.dropna(subset=[INPUT_COL, TARGET_COL])
print(f"✅ Clean dataset: {len(df)} rows (after removing NaN)\n")


# 🎯 SAMPLE DATA FOR FASTER TRAINING
SAMPLE_SIZE = 500  # Use 500 examples for 30-min training
EPOCHS = 50  # 50 epochs - good balance for quality


if len(df) > SAMPLE_SIZE:
    print(f"⚡ Sampling {SAMPLE_SIZE} examples from {len(df)} total rows")
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"✅ Using {len(df)} sampled examples with {EPOCHS} epochs")
    print(f"⏱️  Estimated training time: ~25-30 minutes\n")
else:
    print(f"✅ Using all {len(df)} examples with {EPOCHS} epochs\n")


def format_example(x):
    return {
        "input_text": str(x[INPUT_COL]).strip(),
        "target_text": str(x[TARGET_COL]).strip()
    }


dataset = Dataset.from_list([format_example(x) for x in df.to_dict(orient="records")])


# ======================= 5. TOKENIZATION ==================
print("🔄 Tokenizing dataset...")


def tokenize(ex):
    inp = tokenizer(
        ex["input_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    lab = tokenizer(
        ex["target_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


    inp["labels"] = lab["input_ids"]
    return inp


tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
print(f"✅ Tokenization complete: {len(tokenized)} examples\n")


# ======================= 6. APPLY LORA ====================
print("🔧 Applying LoRA configuration...")


lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)


model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ======================= 7. TRAIN =========================
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    logging_steps=50,
    save_steps=5000,
    save_total_limit=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
    save_strategy="epoch",
    warmup_steps=50,
    weight_decay=0.01,
    dataloader_num_workers=2,
    load_best_model_at_end=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)


print("\n🚀 Training Started...\n")
trainer.train()


# ======================= 8. SAVE MODEL ====================
print("\n💾 Saving model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)


# Save column mapping for inference
config_info = {
    "input_column": INPUT_COL,
    "output_column": TARGET_COL,
    "num_examples": len(df),
    "model_type": "google/flan-t5-base",
    "max_input_length": 256,
    "max_output_length": 256
}


with open(os.path.join(output_dir, "training_config.json"), "w") as f:
    json.dump(config_info, f, indent=2)


print("\n✅ FINETUNING COMPLETE!")
print(f"📁 Model saved to: {output_dir}")
print(f"🎯 Trained on {len(df)} examples using columns: {INPUT_COL} → {TARGET_COL}")
print(f"💡 Ready for production use with FastAPI backend!")