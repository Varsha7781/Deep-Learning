import re
import numpy as np

class SpeechMetricService:
    """
    Extracts high-level speech metrics from Whisper transcription segments.
    """
    def __init__(self):
        # Common English filler words & phrases
        self.filler_patterns = [
            r"\bumm?\b", r"\buhh?\b", r"\bhmm?\b", r"\bahh?\b",
            r"\blike\b", r"\byou know\b", r"\bactually\b", 
            r"\bbasically\b", r"\bso\b", r"\bi mean\b"
        ]

    def extract_metrics(self, transcript_segments, duration_seconds: float):
        """
        Processes Whisper segments to calculate metrics.
        """
        if not transcript_segments:
            return {
                "word_count": 0, "wpm": 0,
                "filler_words": {"total": 0, "breakdown": {}},
                "pauses": {"total_count": 0, "total_duration": 0, "details": []}
            }

        full_text = " ".join([seg["text"] for seg in transcript_segments]).strip().lower()
        words = re.findall(r"\b\w+\b", full_text)
        word_count = len(words)
        
        # 1. Speech Rate (WPM)
        duration_minutes = duration_seconds / 60
        wpm = (word_count / duration_minutes) if duration_minutes > 0 else 0
        
        # 2. Filler Word Detection (using regex patterns)
        fillers_found = {}
        total_fillers = 0
        for pattern in self.filler_patterns:
            matches = re.findall(pattern, full_text)
            count = len(matches)
            if count > 0:
                # Use the pattern as key (cleaned up)
                key = pattern.replace(r"\b", "").replace("?", "")
                fillers_found[key] = count
                total_fillers += count

        # 3. Pause Detection
        # Analyze gaps between sequential segments
        pauses = []
        for i in range(len(transcript_segments) - 1):
            gap = transcript_segments[i+1]["start"] - transcript_segments[i]["end"]
            if gap > 0.6: # 0.6s is a clear intentional or awkward pause
                pauses.append({
                    "start": round(transcript_segments[i]["end"], 2),
                    "duration": round(gap, 2)
                })

        return {
            "word_count": word_count,
            "wpm": round(wpm, 1),
            "filler_words": {
                "total": total_fillers,
                "breakdown": fillers_found
            },
            "pauses": {
                "total_count": len(pauses),
                "total_duration": round(sum(p["duration"] for p in pauses), 2),
                "details": pauses
            }
        }
