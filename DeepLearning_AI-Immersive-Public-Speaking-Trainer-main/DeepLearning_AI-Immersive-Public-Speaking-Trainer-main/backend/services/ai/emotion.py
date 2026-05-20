import os
import torch
import torch.nn as nn
import librosa
import numpy as np

class SpeechEmotionCNN_LSTM(nn.Module):
    """
    CNN + LSTM Architecture for Speech Emotion Recognition.
    - CNN layers extract spatial features from Mel Spectrograms.
    - LSTM layers captures temporal dependencies across frames.
    """
    def __init__(self, num_classes=5): # Happy, Sad, Angry, Fearful, Neutral
        super(SpeechEmotionCNN_LSTM, self).__init__()
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # The input_size depends on the frequency bins after pooling
        self.lstm = nn.LSTM(input_size=128 * 32, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch, 1, frequency_bins, time_frames)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        
        # Reshape for LSTM: (batch, time, features)
        batch_size, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time, channels * freq)
        
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :]) # Use last hidden state
        return self.softmax(x)

class EmotionDetectionService:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpeechEmotionCNN_LSTM(num_classes=5).to(self.device)
        self.emotions = ["Happy", "Sad", "Angry", "Fearful", "Neutral"]
        self.is_loaded = False
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.is_loaded = True
                print(f"Emotion model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading emotion model: {e}")
        
        self.model.eval()

    def _extract_mfcc(self, audio_path, n_mfcc=128, max_len=128):
        """
        Extract MFCC features from raw audio.
        Returns a normalized spectrogram of shape (n_mfcc, max_len).
        """
        y, sr = librosa.load(audio_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Fixed length padding/cropping
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        
        # Normalize to Z-score
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        return mfcc

    def detect_emotion(self, audio_path: str):
        """
        Analyze audio and predict the speaker's emotion.
        If no model is loaded, it defaults to 'Neutral' with moderate confidence.
        """
        if not self.is_loaded:
            print("⚠ Emotion model not loaded — using fallback")
            return {
                "prediction": "Neutral",
                "confidence": 0.85,
                "probabilities": {e: 0.2 for e in self.emotions},
                "status": "No model loaded, using fallback"
            }

        try:
            mfcc = self._extract_mfcc(audio_path)
            input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = output[0].tolist()
                max_idx = np.argmax(probs)
                
            return {
                "prediction": self.emotions[max_idx],
                "confidence": round(probs[max_idx], 4),
                "probabilities": dict(zip(self.emotions, [round(p, 4) for p in probs]))
            }
        except Exception as e:
            print(f"Emotion Detection Error: {e}")
            return {"prediction": "Neutral", "confidence": 0.5, "error": str(e)}
