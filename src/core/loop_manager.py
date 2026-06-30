import re
from .detector import EnsembleJudge, LocalDetector
from .generator import LLMClient

class HumanizerLoop:
    def __init__(self, llm_client, judge=None):
        self.llm = llm_client
        self.judge = judge or EnsembleJudge()

    def split_into_sentences(self, text):
        return re.split(r'(?<=[.!?])\s+', text)

    def run(self, initial_text, threshold=80, max_iterations=5, persona="professional", history_size=2):
        current_text = initial_text
        iteration = 0
        import random
        
        # Track history per sentence: {sentence_index: [(text, score), ...]}
        sentence_history = {}
        
        while iteration < max_iterations:
            probs = self.judge.local.get_sentence_probabilities(current_text)
            sentences = re.split(r'(?<=[.!?])\s+', current_text)
            
            humanity_score = 0
            if probs:
                human_like_count = sum(1 for p in probs if p < 0.5)
                humanity_score = (human_like_count / len(probs)) * 100
            
            if humanity_score >= threshold:
                current_text = self.llm.polish_cohesion(current_text)
                return current_text, humanity_score, iteration
            
            iteration += 1
            
            new_sentences = []
            for i, s in enumerate(sentences):
                # Only rewrite if it's flagged as AI (>60%)
                if i < len(probs) and probs[i] > 0.6:
                    template = random.choice(["punchy", "sprawl", "inverted"])
                    
                    # Best-of-N Sampling (N=3)
                    best_s = s
                    best_score = probs[i] # Start with current AI prob
                    
                    history = sentence_history.get(i, [])
                    
                    for _ in range(3):
                        candidate = self.llm.reconstruct_sentence(
                            s, template, 
                            temperature=0.8 + (iteration * 0.1), 
                            history=history
                        )
                        # Get AI prob for candidate
                        cand_prob = self.judge.local.get_sentence_probabilities(candidate)[0] if candidate else 1.0
                        if cand_prob < best_score:
                            best_score = cand_prob
                            best_s = candidate
                    
                    # Update history for this sentence (keep last N)
                    history_entry = (best_s, (1.0 - best_score) * 100)
                    sentence_history[i] = (sentence_history.get(i, []) + [history_entry])[-history_size:]
                    s = best_s
                new_sentences.append(s)
            
            current_text = " ".join(new_sentences)
            
        current_text = self.llm.polish_cohesion(current_text)
        final_probs = self.judge.local.get_sentence_probabilities(current_text)
        final_score = 0
        if final_probs:
            human_like_count = sum(1 for p in final_probs if p < 0.5)
            final_score = (human_like_count / len(final_probs)) * 100
        return current_text, final_score, iteration
            
        final_score, _ = self.judge.analyze(current_text)
        return current_text, final_score, iteration
