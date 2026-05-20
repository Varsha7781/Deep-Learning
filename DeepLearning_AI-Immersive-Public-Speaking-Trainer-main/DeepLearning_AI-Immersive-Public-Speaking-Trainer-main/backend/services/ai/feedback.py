class CoachFeedbackService:
    """
    Generates actionable coaching feedback based on speech and emotion metrics.
    """
    def generate_feedback(self, speech_rate, filler_words, confidence_score, emotion):
        """
        Generates deterministic rule-based feedback.
        """
        strengths = []
        weaknesses = []
        suggestions = []

        # 1. Speech Rate Logic
        if 120 <= speech_rate <= 160:
            strengths.append("Good speaking pace")
        elif speech_rate < 100:
            weaknesses.append("Speech is too slow")
            suggestions.append("Try to increase your speaking speed for better engagement")
        elif speech_rate > 170:
            weaknesses.append("Speech is too fast")
            suggestions.append("Slow down to allow your audience to process the information")

        # 2. Filler Word Logic
        if filler_words <= 2:
            strengths.append("Clear and fluent speech")
        elif 3 <= filler_words <= 6:
            weaknesses.append("Moderate use of filler words")
            suggestions.append("Practice replacing filler words with brief pauses")
        else:
            weaknesses.append("High filler word usage")
            suggestions.append("Focus on intentional pausing to eliminate 'um's and 'uh's")

        # 3. Emotion Logic
        if emotion in ["Happy", "Neutral"]:
            strengths.append("Confident tone")
        elif emotion == "Nervous":
            weaknesses.append("Slight nervousness detected")
            suggestions.append("Focus on steady breathing to project more authority")
        elif emotion == "Sad":
            weaknesses.append("Low energy delivery")
            suggestions.append("Try to vary your pitch to keep the presentation dynamic")

        # 4. Confidence Score Logic
        if confidence_score > 80:
            strengths.append("Strong overall performance")
        elif 50 <= confidence_score <= 80:
            suggestions.append("Practice more for consistency")
        else:
            suggestions.append("Needs significant improvement")

        # Final Polish: Ensure at least 1 in each
        if not strengths:
            strengths.append("Consistent attempt at professional delivery")
        if not weaknesses:
            weaknesses.append("No major technical issues detected")
        if not suggestions:
            suggestions.append("Continue practicing to maintain this level of performance")

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions
        }
