import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import re

class LocalDetector:
    def __init__(self, model_name="gpt2", classifier_name="roberta-base-openai-detector"):
        # Perplexity Model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        # Neural Classifier Model
        self.clf_tokenizer = AutoTokenizer.from_pretrained(classifier_name)
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(classifier_name)
        self.clf_model.eval()

    def calculate_perplexity(self, text):
        if not text or len(text.strip()) == 0:
            return 0
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings["input_ids"])
            return torch.exp(outputs.loss).item()

    def get_sentence_probabilities(self, text):
        """Returns a list of AI probabilities for each sentence in the text."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        probs = []
        for s in sentences:
            if not s.strip():
                probs.append(0.0)
                continue
            inputs = self.clf_tokenizer(s, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = self.clf_model(**inputs).logits
                probs_tensor = torch.softmax(logits, dim=-1)
                probs.append(probs_tensor[0][1].item())
        return probs

    def get_humanity_score(self, text):
        """Returns a score from 0-100 where 100 is most human."""
        probs = self.get_sentence_probabilities(text)
        if not probs:
            return 0.0
        avg_ai_prob = sum(probs) / len(probs)
        return (1.0 - avg_ai_prob) * 100

class APIDetector:
    def __init__(self, api_key=None, provider="generic"):
        self.api_key = api_key
        self.provider = provider

    def get_score(self, text):
        if not self.api_key: return None
        return 50 # Mock

class EnsembleJudge:
    def __init__(self, local_detector=None, api_detectors=None):
        self.local = local_detector or LocalDetector()
        self.apis = api_detectors or []

    def analyze(self, text):
        humanity_score = self.local.get_humanity_score(text)
        api_scores = [api.get_score(text) for api in self.apis if api.get_score(text) is not None]
        if not api_scores:
            return humanity_score, humanity_score
        avg_score = (humanity_score + sum(api_scores)) / (1 + len(api_scores))
        return avg_score, humanity_score
