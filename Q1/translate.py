from transformers import MarianMTModel, MarianTokenizer
from striprtf.striprtf import rtf_to_text
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

INPUT_FILE = "input.rtf"
REF_FILE = "output.rtf"
OUTPUT_FILE = "output.txt"

def read_rtf(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        rtf_content = f.read()
    text = rtf_to_text(rtf_content)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[1:]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-bn-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-bn-en")
    return tokenizer, model


def translate_lines(lines, tokenizer, model):
    translated = []
    for line in lines:
        inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated.append(translated_text)
    return translated


def save_output(lines):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def compute_bleu(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def main():
    print("Loading input...")
    input_lines = read_rtf(INPUT_FILE)

    print("Loading reference...")
    ref_lines = read_rtf(REF_FILE)

    print("Loading model...")
    tokenizer, model = load_model()

    print("Translating...")
    translated_lines = translate_lines(input_lines, tokenizer, model)

    print("Saving output...")
    save_output(translated_lines)

    print("\nComputing BLEU score...")
    bleu_score = compute_bleu(translated_lines, ref_lines)

    print(f"\nBLEU Score: {bleu_score:.2f}")


if __name__ == "__main__":
    main()