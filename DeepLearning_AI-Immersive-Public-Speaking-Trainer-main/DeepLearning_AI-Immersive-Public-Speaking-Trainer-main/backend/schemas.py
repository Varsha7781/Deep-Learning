from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator
import re


# ── Request Schemas ──────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    confirm_password: str

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


# ── Response Schemas ─────────────────────────────────────────

class UserResponse(BaseModel):
    name: str
    email: str

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── AI Analysis Schemas ──────────────────────────────────────

class EmotionResult(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]

class FillerWords(BaseModel):
    total: int
    breakdown: dict[str, int]

class PauseDetail(BaseModel):
    start: float
    duration: float

class Pauses(BaseModel):
    total_count: int
    total_duration: float
    details: list[PauseDetail]

class SpeechMetrics(BaseModel):
    word_count: int
    wpm: float
    filler_words: FillerWords
    pauses: Pauses

class CoachingFeedback(BaseModel):
    overall_summary: str
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    practice_tips: list[str]

class AnalysisResponse(BaseModel):
    transcript: str
    emotion: EmotionResult
    metrics: SpeechMetrics
    confidence_score: float
    feedback: CoachingFeedback
    duration_seconds: float

# ── Simplified AI Analysis Schemas ──────────────────────────

class SimpleMetrics(BaseModel):
    word_count: int
    speech_rate: float
    filler_words: int

class SimpleEmotionResponse(BaseModel):
    label: str
    confidence: float

class SimpleCoachingFeedback(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]

class SimplifiedAnalysisResponse(BaseModel):
    id: Optional[str] = None
    transcription: str
    duration: float
    confidence_score: float
    emotion: SimpleEmotionResponse
    metrics: SimpleMetrics
    feedback: SimpleCoachingFeedback
    created_at: Optional[datetime] = None

class HistoryItem(BaseModel):
    id: str
    date: datetime
    title: str
    score: float
    duration: float
    emotion: str

class DashboardStats(BaseModel):
    avg_confidence: float
    total_speeches: int
    total_hours: float
    improvement: float
