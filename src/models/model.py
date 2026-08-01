from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_model(model_name: str, num_labels: int = 3):
    """
    Load pre-trained multilingual transformer model for sequence classification.
    """
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def get_tokenizer(model_name: str):
    """
    Load tokenizer associated with the model.
    """
    return AutoTokenizer.from_pretrained(model_name)
