# ===================== HEALTHCARE CHATBOT - INFERENCE CODE WITH VOICE SUPPORT =====================
# Updated inference code that includes voice input processing
# This code loads the trained model and prepares it for use with FastAPI backend

# ===================== 1. INSTALL =====================
# !pip install -q transformers peft accelerate sentencepiece


# ===================== 2. MOUNT DRIVE ==================
from google.colab import drive
drive.mount('/content/drive')


# ===================== 3. IMPORTS =====================
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


# ===================== 4. PATHS =======================
model_path = "/content/drive/MyDrive/4-2 project/google_flan-t5-base-trained"


# Check what files exist in the directory
print("📁 Checking saved files...")
if os.path.exists(model_path):
    files = os.listdir(model_path)
    print(f"✅ Found files: {files}\n")
else:
    print(f"❌ Directory not found: {model_path}")
    raise FileNotFoundError("Model directory doesn't exist!")


# Check if LoRA adapter files exist
has_adapter = "adapter_config.json" in files
has_model = "adapter_model.bin" in files or "adapter_model.safetensors" in files


print(f"LoRA adapter config: {'✅' if has_adapter else '❌'}")
print(f"LoRA adapter weights: {'✅' if has_model else '❌'}\n")


# ===================== 5. LOAD MODEL ==================
if has_adapter and has_model:
    # Method 1: Load with LoRA adapter (if training completed successfully)
    print("🔄 Loading base model...")
    base_model = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)


    print("🔄 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    print("✅ Model loaded with LoRA adapter!\n")


else:
    # Method 2: Load directly from saved checkpoint (fallback)
    print("⚠️ LoRA adapter not found. Loading from checkpoint...")


    # Find checkpoint folders
    checkpoints = [f for f in files if f.startswith("checkpoint-")]


    if checkpoints:
        # Use the latest checkpoint
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        checkpoint_path = os.path.join(model_path, latest_checkpoint)
        print(f"📂 Using checkpoint: {latest_checkpoint}")


        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        print("✅ Model loaded from checkpoint!\n")
    else:
        # Load base model as last resort
        print("⚠️ No checkpoints found. Loading base FLAN-T5 model...")
        base_model = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        print("⚠️ WARNING: Using untrained base model!\n")


# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()


print(f"🖥️  Using device: {device}")
print("=" * 50)


# ===================== 6. PREDICT FUNCTION =============
def ask(question, max_length=256, temperature=0.7, num_beams=4):
    """
    Generate a response to a healthcare question
    Works with both text and transcribed voice input

    Args:
        question: User's input question (text or transcribed voice)
        max_length: Maximum response length
        temperature: Sampling temperature (higher = more creative)
        num_beams: Beam search size (higher = better quality but slower)
    
    Returns:
        str: Generated response
    """
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)


    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            num_beams=num_beams,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )


    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# ===================== 7. VOICE PROCESSING FUNCTION =============
def process_voice_input(transcribed_text: str) -> str:
    """
    Process transcribed voice input and return AI response
    
    Args:
        transcribed_text: Text transcribed from user's voice input
    
    Returns:
        str: AI chatbot's response
    """
    # Clean the transcribed text
    cleaned_text = transcribed_text.strip()
    
    if not cleaned_text:
        return "I didn't catch that. Please try again."
    
    # Get response from model
    response = ask(cleaned_text)
    
    return response


# ===================== 8. VOICE + TEXT RESPONSE GENERATOR =============
def process_input(input_text: str, is_voice: bool = False) -> dict:
    """
    Process user input (voice or text) and return structured response
    
    Args:
        input_text: User input (transcribed voice or typed text)
        is_voice: Boolean indicating if input is from voice
    
    Returns:
        dict: Contains response and metadata
    """
    try:
        response = ask(input_text)
        
        return {
            "status": "success",
            "input": input_text,
            "input_type": "voice" if is_voice else "text",
            "response": response,
            "confidence": 0.95  # Placeholder confidence score
        }
    
    except Exception as e:
        return {
            "status": "error",
            "input": input_text,
            "error": str(e),
            "response": "Sorry, I encountered an error processing your request."
        }


# ===================== 9. TEST THE MODEL ===============
print("\n🧪 Testing model with sample questions...\n")


test_questions = [
    "What are the symptoms of diabetes?",
    "How can I treat a cold at home?"
]


for i, question in enumerate(test_questions, 1):
    print(f"Q{i}: {question}")
    response = ask(question)
    print(f"A{i}: {response}\n")


# ===================== 10. TEST VOICE PROCESSING ===============
print("\n🎤 Testing voice processing...\n")


voice_test_inputs = [
    "What should I do for headache",
    "Tell me about asthma symptoms"
]


for i, voice_input in enumerate(voice_test_inputs, 1):
    print(f"Voice Input {i}: {voice_input}")
    result = process_input(voice_input, is_voice=True)
    print(f"Response: {result['response']}\n")


# ===================== 11. INTERACTIVE CHAT WITH VOICE SUPPORT =============
print("=" * 50)
print("🤖 Healthcare Chatbot Ready!")
print("   ✅ Text Input Support")
print("   ✅ Voice Input Support (via FastAPI)")
print("=" * 50)
print("\nTips:")
print("  • Type your health question or use voice input")
print("  • Voice input works through the FastAPI endpoint")
print("  • Type 'exit' or 'quit' to stop")
print("  • Type 'help' for usage tips\n")


while True:
    try:
        user_input = input("You: ").strip()


        if not user_input:
            continue


        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\n👋 Goodbye! Stay healthy!")
            break


        if user_input.lower() == 'help':
            print("\n💡 Usage Tips:")
            print("  • Describe your symptoms clearly")
            print("  • Ask about specific conditions or medications")
            print("  • Request health advice or home remedies")
            print("  • Always consult a doctor for serious issues!")
            print("  • Voice input: Use the web interface for voice\n")
            continue


        response = ask(user_input)
        print(f"Bot: {response}\n")


    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Stay healthy!")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please try again.\n")


# ===================== 12. EXPORT FOR PRODUCTION =============
print("\n📦 Model is ready for production use with FastAPI!")
print("   • Model loaded and tested")
print("   • Voice processing functions available")
print("   • Ready to integrate with FastAPI backend")