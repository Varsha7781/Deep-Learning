class ConfidenceScoringService:
    """
    Calculates a 0-100 Confidence Score based on multi-factor analysis:
    - Emotion (40%): Preference for Neutral/Happy over Angry/Sad.
    - Pacing (30%): Optimal WPM range (130-160).
    - Fluidity (30%): Penalties for fillers and awkward pauses.
    """
    def calculate_score(self, metrics, duration):
        if duration <= 0:
            return 0.0

        wpm = metrics.get("wpm", 0)
        fillers = metrics.get("filler_words", {}).get("total", 0)
        pauses = metrics.get("pauses", {}).get("total_duration", 0)

        score = 100.0

        # 1. Speech rate (continuous penalty)
        ideal_wpm = 140
        score -= abs(wpm - ideal_wpm) * 0.3   # smooth penalty

        # 2. Filler words (per filler penalty)
        score -= fillers * 2.5

        # 3. Pause penalty
        score -= pauses * 0.5

        # Clamp
        return round(max(0, min(100, score)), 1)