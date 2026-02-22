from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

HF_USERNAME = "arpita2desh"
REPO_NAME = "distilbert-genres-mlops"

LOCAL_MODEL_PATH = "distilbert-reviews-genres"
FULL_REPO_NAME = f"{HF_USERNAME}/{REPO_NAME}"

print("Loading model from local folder...")
model = DistilBertForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
tokenizer = DistilBertTokenizerFast.from_pretrained(LOCAL_MODEL_PATH)

print("Uploading to Hugging Face Hub...")
model.push_to_hub(FULL_REPO_NAME)
tokenizer.push_to_hub(FULL_REPO_NAME)

print("Upload complete!")
