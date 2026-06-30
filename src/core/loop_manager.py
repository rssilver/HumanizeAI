import re
from .detector import EnsembleJudge, LocalDetector
from .generator import LLMClient

class HumanizerLoop:
    def __init__(self, llm_client, judge=None):
        self.llm = llm_client
        self.judge = judge or EnsembleJudge()

    def split_into_sentences(self, text):
        return re.split(r'(?<=[.!?])\s+', text)

    def run(self, initial_text, threshold=80, max_iterations=5, persona="professional"):
        current_text = initial_text
        iteration = 0
        import random
        
        while iteration < max_iterations:
            probs = self.judge.local.get_sentence_probabilities(current_text)
            sentences = re.split(r'(?<=[.!?])\s+', current_text)
            
            avg_ai_prob = sum(probs) / len(probs) if probs else 1.0
            humanity_score = (1.0 - avg_ai_prob) * 100
            
            if humanity_score >= threshold:
                # Final polish pass to fix readability before returning
                current_text = self.llm.polish_cohesion(current_text)
                return current_text, humanity_score, iteration
            
            iteration += 1
            
            new_sentences = []
            for i, s in enumerate(sentences):
                if i < len(probs) and probs[i] > 0.6:
                    template = random.choice(["punchy", "sprawl", "inverted"])
                    s = self.llm.reconstruct_sentence(s, template, temperature=0.7 + (iteration * 0.1))
                new_sentences.append(s)
            
            current_text = " ".join(new_sentences)
            
        # If we hit max iterations without hitting threshold, still apply a final polish for readability
        current_text = self.llm.polish_cohesion(current_text)
        final_probs = self.judge.local.get_sentence_probabilities(current_text)
        final_score = (1.0 - (sum(final_probs) / len(final_probs))) * 100 if final_probs else 0
        return current_text, final_score, iteration
            
        final_score, _ = self.judge.analyze(current_text)
        return current_text, final_score, iteration
