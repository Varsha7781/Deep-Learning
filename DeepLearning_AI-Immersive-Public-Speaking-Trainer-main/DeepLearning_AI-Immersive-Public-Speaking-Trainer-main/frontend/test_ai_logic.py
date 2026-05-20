import sys
import unittest.mock as mock

# Mock heavy dependencies before they are imported by backend services
sys.modules['whisper'] = mock.MagicMock()
sys.modules['torch'] = mock.MagicMock()
sys.modules['torch.nn'] = mock.MagicMock()
sys.modules['librosa'] = mock.MagicMock()
sys.modules['librosa.feature'] = mock.MagicMock()

import os
# Add backend directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from backend.services.ai.metrics import SpeechMetricService
from backend.services.ai.scoring import ConfidenceScoringService
from backend.services.ai.feedback import CoachFeedbackService

def test_ai_logic():
    print("--- Testing AI Coaching Logic (Metrics, Scoring, Feedback) ---")
    
    metric_service = SpeechMetricService()
    scoring_service = ConfidenceScoringService()
    feedback_service = CoachFeedbackService()
    
    # Mock Whisper segments
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello everyone today I want to talk about"},
        {"start": 2.5, "end": 5.0, "text": "Umm like you know the future of AI"},
        {"start": 7.0, "end": 10.0, "text": "It is actually very basically amazing."}
    ]
    duration = 10.0 # seconds
    
    # 1. Test Metrics
    print("\n1. Testing Metrics...")
    metrics = metric_service.extract_metrics(segments, duration)
    print(f"Word Count: {metrics['word_count']}")
    print(f"WPM: {metrics['wpm']}")
    print(f"Fillers: {metrics['filler_words']['total']}")
    print(f"Pauses: {metrics['pauses']['total_count']}")
    
    assert metrics['word_count'] > 0
    # Expected fillers: "umm", "like", "you know", "actually", "basically"
    assert metrics['filler_words']['total'] >= 4
    
    # 2. Test Scoring
    print("\n2. Testing Scoring...")
    emotion_result = {"prediction": "Neutral", "confidence": 0.9}
    score = scoring_service.calculate_score(emotion_result, metrics, duration)
    print(f"Confidence Score: {score}")
    
    assert 0 <= score <= 100
    
    # 3. Test Feedback
    print("\n3. Testing Feedback...")
    feedback = feedback_service.generate_feedback(emotion_result, metrics, score)
    print(f"Summary: {feedback['overall_summary']}")
    print(f"Strengths: {len(feedback['strengths'])}")
    print(f"Weaknesses: {len(feedback['weaknesses'])}")
    
    assert "strengths" in feedback
    assert "weaknesses" in feedback
    
    print("\n--- Logic Test Passed! ---")

if __name__ == "__main__":
    test_ai_logic()
