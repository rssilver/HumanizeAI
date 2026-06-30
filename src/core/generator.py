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

    def reconstruct_sentence(self, sentence, template_type, temperature=0.7, history=None):
        """Rebuilds a single sentence using the 'Pattern-Slayer' logic to eliminate AI anchors."""
        templates = {
            "punchy": "Rewrite this as a concise, impactful human observation (under 12 words). Avoid all fluff and robotic summaries.",
            "sprawl": "Rewrite this as a natural, flowing complex sentence (over 20 words) with subordinate clauses. It must sound like an organic thought, not a generated list.",
            "inverted": "Rewrite this starting with an introductory phrase or adverb to shift the focus. It must feel like a natural human transition."
        }
        
        history_context = ""
        if history:
            history_context = "\nPrevious attempts (Avoid these specific phrasings):\n" + "\n".join([f"- {h[0]}" for h in history])

        prompt = (
            f"You are a ruthless professional editor. Your goal is to strip away 'AI-isms'—the predictable structural anchors that detectors use.\n\n"
            f"STRICT TRANSFORMATION RULES:\n"
            f"1. KILL THE SIGNPOSTS: Remove words like 'Furthermore', 'Moreover', 'Consequently', and 'In conclusion'. Start the sentence directly.\n"
            f"2. BREAK SYMMETRY: If a sentence is balanced (While X, Y), break it into asymmetrical parts or change the rhythm.\n"
            f"3. DELETE FLUFF PAIRS: Replace adjective pairs (e.g., 'comprehensive and nuanced') with one precise word or a concrete detail.\n"
            f"4. GROUND THE INTRO: Remove generic openings about 'the modern digital world' or 'the era of X'.\n"
            f"5. SIMPLIFY VERBS: Replace high-frequency AI verbs (underscores, facilitates, delves) with natural alternatives.\n"
            f"6. NO SUMMARY TONE: Avoid wrapping up thoughts with 'Overall' or 'Ultimately'.\n\n"
            f"CONSTRAINT: {templates[template_type]}\n"
            f"{history_context}\n\n"
            f"Original Sentence: {sentence}\n\n"
            f"Provide ONLY the rewritten sentence."
        )
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

    def polish_cohesion(self, text, temperature=0.4):
        """Final pass to restore elegance while strictly avoiding AI-typical 'smoothing' patterns."""
        prompt = (
            f"Act as a world-class editor. The following text has been structurally varied for rhythm. "
            f"Your goal is to make it read elegantly and fluidly, but you MUST avoid the 'AI-smoothness' trap.\n\n"
            f"STRICT CONSTRAINTS:\n"
            f"1. NO AI TRANSITIONS: Absolutely ban words like 'Moreover', 'Furthermore', 'Additionally', or 'In conclusion'.\n"
            f"2. PRESERVE THE JOLT: Keep the contrast between very short and very long sentences. Do not average them out.\n"
            f"3. HUMAN CONNECTORS: Use organic, slightly less predictable transitions (e.g., 'The thing is', 'And yet', 'Truth be told') or simply start the sentence cold.\n"
            f"4. AVOID OVER-POLISHING: If a sentence sounds slightly idiosyncratic but clear, leave it alone. Do not make it 'perfect'.\n\n"
            f"Text:\n{text}"
        )
        return self.generate(prompt, temperature=temperature)
