import requests
import json
import random

class LLMClient:
    def __init__(self, base_url="http://127.0.0.1:1234", model="lmstudio-community/gemma-4-31b-it-uncensored-heretic"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt, temperature=0.7):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }
        try:
            response = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error connecting to LM Studio: {str(e)}"

    def reconstruct_sentence(self, sentence, template_type, temperature=0.7):
        """Rebuilds a single sentence using a specific structural template."""
        templates = {
            "punchy": "Rewrite this sentence to be very short and punchy (under 10 words). Avoid all fluff.",
            "sprawl": "Rewrite this sentence to be long and complex (over 25 words), using subordinate clauses, but keep it natural.",
            "inverted": "Rewrite this sentence so it does NOT start with the subject. Start with a prepositional phrase or an adverb."
        }
        prompt = f"{templates[template_type]}\n\nOriginal: {sentence}\n\nProvide ONLY the rewritten sentence."
        return self.generate(prompt, temperature=temperature)

    def humanize_pipeline(self, text, persona="professional", temperature=0.7):
        """Two-pass transformation pipeline based on selected persona."""
        if persona == "casual":
            deconstruct_prompt = (
                f"Rewrite the following text to sound like a real person talking to a colleague. "
                f"Use natural, slightly irregular phrasing. Use occasional sentence fragments. "
                f"Avoid perfectly balanced lists or corporate-speak. Keep it grounded and authentic.\n\n"
                f"Text:\n{text}"
            )
            messy_text = self.generate(deconstruct_prompt, temperature=temperature + 0.2)
            recompose_prompt = (
                f"Clean up the grammar and flow of the following text to make it professional yet human. "
                f"CRITICAL: Preserve the irregular sentence lengths, idiosyncratic phrasing, and natural rhythm "
                f"of the input. Do not make it sound like a textbook or an AI assistant.\n\n"
                f"Text:\n{messy_text}"
            )
            return self.generate(recompose_prompt, temperature=temperature)
        else:
            deconstruct_prompt = (
                f"Rewrite the following text to eliminate predictable AI patterns. "
                f"CRITICAL INSTRUCTIONS:\n"
                f"1. VARY SENTENCE LENGTH: Mix very short, punchy sentences with longer, complex ones.\n"
                f"2. REMOVE FILLER: Eliminate conversational fluff (e.g., 'It is wild that...').\n"
                f"3. AVOID AI CLICHÉS: Do not use words like 'tapestry', 'testament', 'delve', or 'moreover'.\n"
                f"4. NATURAL SYNTAX: Use a professional but direct tone. Avoid perfectly balanced triplets.\n\n"
                f"Text:\n{text}"
            )
            diversified_text = self.generate(deconstruct_prompt, temperature=temperature + 0.2)
            recompose_prompt = (
                f"Refine the following text for professional clarity while strictly preserving "
                f"the irregular sentence lengths and natural rhythm of the input. "
                f"Do not 'smooth out' the variance.\n\n"
                f"Text:\n{diversified_text}"
            )
            return self.generate(recompose_prompt, temperature=temperature)

    def polish_cohesion(self, text, temperature=0.5):
        """Final pass to ensure flow and readability without losing structural variance."""
        prompt = (
            f"The following text has been structurally varied for rhythm. "
            f"Please refine it for natural flow and cohesion. "
            f"CRITICAL: Do not 'smooth out' the sentence lengths. Keep the short punchy sentences "
            f"and the long complex ones exactly where they are, but ensure the transitions between them "
            f"are smooth and grammatically correct.\n\n"
            f"Text:\n{text}"
        )
        return self.generate(prompt, temperature=temperature)
