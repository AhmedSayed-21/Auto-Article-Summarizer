

```markdown
# 📰 Text Summarization with DistilBART + LoRA

This project fine-tunes **DistilBART (`sshleifer/distilbart-cnn-12-6`)** on the **CNN/DailyMail dataset** using **LoRA (Low-Rank Adaptation)** to create an efficient and accurate text summarization model.  
It supports **GPU/CPU inference**, zero-shot/few-shot prompting, and saves the final model for deployment.

---

## 🚀 Features

- ✅ Load and preprocess **CNN/DailyMail 3.0.0** dataset.  
- ✅ Tokenize and truncate long articles into model-compatible sequences.  
- ✅ Fine-tune DistilBART using **LoRA for parameter-efficient training**.  
- ✅ Save final model (`./summarizer_lora_final`).  
- ✅ Run summarization on **GPU** or **CPU**.  
- ✅ Generate summaries with **zero-shot** and **few-shot** prompts.  

---

## 📂 Project Structure

```

summarizer_lora.ipynb        # Main training & inference notebook
summarizer_lora_final/       # Fine-tuned model (saved)

````

---

## ⚙️ Installation

Install dependencies with:

```bash
pip install torch transformers datasets accelerate peft
````

---

## 📊 Dataset

The project uses the **CNN/DailyMail summarization dataset**:

* **article** → input news article
* **highlights** → target summary

During training, articles are truncated to **512 tokens** and summaries to **128 tokens**.

---

## 🏋️ Training

Training is done with HuggingFace `Seq2SeqTrainer` and LoRA:

```python
trainer.train()
```

The trained model will be saved under:

```
./summarizer_lora_final
```

---

## ✅ Inference

### Example: Summarization with the trained model

```python
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="./summarizer_lora_final",
    tokenizer=tokenizer,
    device=0  # set to -1 for CPU
)

article = "Your long article text here..."
print(summarizer(article, max_length=150, min_length=40, do_sample=False)[0]["summary_text"])
```

---

## 🧪 Prompting Examples

### Zero-shot

```
Summarize the following article into key points:
<Article Text>
```

### Few-shot

```
Example:
Article: Artificial Intelligence has rapidly evolved in the 21st century. It is now used in many industries including healthcare and finance.
Summary: AI has evolved quickly and is now used across multiple industries.

Now summarize the following article into key points:
<Article Text>
```

---

## 🔧 Future Work

* Extend to multi-document summarization.
* Add evaluation metrics (ROUGE, BLEU).
* Convert notebook into Python scripts for CLI usage.

---

## 📜 License

MIT License – feel free to use, modify, and share.

```

---

```
