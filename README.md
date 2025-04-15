
# AI Book Code Examples

This repository contains hands-on examples from Chapter 10 and Chapter 11 of the book:
**AI Product Development & Ethics: A Practical Fast-track Handbook for Product Managers and Developers**

## Chapters

### Chapter 10: Hands-on with AI
This section includes code to:
- Set up local/cloud environments
- Train a simple neural network using PyTorch on MNIST
- Deploy the model using FastAPI

### Chapter 11: Building Simple AI Models
This section covers:
- Fine-tuning a DistilBERT model for sentiment classification using the IMDB dataset

## File Structure
```
ai_book_code/
├── 01_setup_environment.md
├── 02_train_simple_model.py
├── 03_deploy_model_fastapi.py
└── 04_build_text_classifier.py
```

## Requirements
- Python 3.8+
- PyTorch, torchvision
- transformers, datasets
- fastapi, uvicorn

## Getting Started

To train the MNIST model:
```bash
python 02_train_simple_model.py
```

To run the FastAPI server:
```bash
uvicorn 03_deploy_model_fastapi:app --reload
```

To fine-tune the DistilBERT model:
```bash
python 04_build_text_classifier.py
```

## License
MIT
