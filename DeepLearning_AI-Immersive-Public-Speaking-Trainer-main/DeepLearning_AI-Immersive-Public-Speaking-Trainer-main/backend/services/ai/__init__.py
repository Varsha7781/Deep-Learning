from .transcription import TranscriptionService
from .emotion import EmotionDetectionService
from .metrics import SpeechMetricService
from .scoring import ConfidenceScoringService
from .feedback import CoachFeedbackService

# Initialize all services once for use in the API
transcription_service = TranscriptionService()
emotion_service = EmotionDetectionService()
metric_service = SpeechMetricService()
scoring_service = ConfidenceScoringService()
feedback_service = CoachFeedbackService()
