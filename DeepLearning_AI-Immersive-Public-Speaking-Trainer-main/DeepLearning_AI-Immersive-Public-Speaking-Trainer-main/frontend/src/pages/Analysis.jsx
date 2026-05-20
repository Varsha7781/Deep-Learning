import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useLocation, Link } from 'react-router-dom';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Radar, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
} from 'recharts';
import { 
  TrendingUp, 
  Award, 
  Clock, 
  Zap, 
  MessageSquare, 
  ShieldCheck, 
  AlertTriangle,
  ArrowLeft,
  LayoutDashboard,
  CheckCircle2,
  AlertCircle,
  Lightbulb
} from 'lucide-react';
import DashboardLayout from '../layouts/DashboardLayout';
import './Analysis.css';

export default function Analysis() {
  const location = useLocation();
  const analysisData = location.state;

  if (!analysisData) {
    return (
      <DashboardLayout>
        <div className="no-data-container glass-panel">
          <AlertTriangle size={64} color="#f59e0b" />
          <h2>No Analysis Data Found</h2>
          <p className="text-secondary">Please complete a practice session to see your results.</p>
          <div style={{ marginTop: '24px' }}>
            <Link to="/practice" className="primary-hud-btn start-btn" style={{ padding: '12px 24px', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
               <ArrowLeft size={20} />
               <span>Go back to Practice</span>
            </Link>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  const { transcription, duration, metrics, confidence_score, emotion, feedback } = analysisData;
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedScore(confidence_score), 500);
    return () => clearTimeout(timer);
  }, [confidence_score]);

  const getScoreColor = (s) => {
    if (s >= 80) return '#10b981'; // Green
    if (s >= 50) return '#f59e0b'; // Amber
    return '#ef4444'; // Red
  };

  const getEmotionColor = (label) => {
    switch (label) {
      case 'Happy': return '#10b981';
      case 'Neutral': return '#3b82f6';
      case 'Nervous': return '#f59e0b';
      case 'Sad': return '#64748b';
      default: return '#94a3b8';
    }
  };

  const formatDuration = (s) => {
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Staggering Variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.2 }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { duration: 0.5, ease: "easeOut" } }
  };

  return (
    <DashboardLayout>
      <motion.div 
        className="analysis-container"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        
        <header className="page-header" style={{ marginBottom: '8px' }}>
           <div className="header-text">
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                <Link to="/practice" className="text-muted" style={{ textDecoration: 'none' }}>
                  <ArrowLeft size={18} />
                </Link>
                <h1 className="display-title" style={{ fontSize: '28px' }}>Session Insights</h1>
              </div>
              <p className="text-secondary" style={{ fontSize: '14px' }}>
                {new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })} • {formatDuration(duration)} duration
              </p>
           </div>
        </header>

        <div className="analysis-grid">
           
           {/* HERO CARD - Confidence Score */}
           <motion.section 
             variants={itemVariants} 
             className="hero-card glass-panel"
             style={{ borderTop: `4px solid ${getScoreColor(confidence_score)}` }}
           >
              <span className="hero-label">Confidence Score</span>
              <div className="hero-score-wrapper">
                 <motion.div 
                   className="hero-score"
                   initial={{ scale: 0.5, opacity: 0 }}
                   animate={{ scale: 1, opacity: 1 }}
                   transition={{ delay: 0.3, type: "spring", stiffness: 100 }}
                 >
                    {Math.round(animatedScore)}
                 </motion.div>
                 
                 {/* Subtle Radial Glow */}
                 <div style={{ 
                   position: 'absolute', 
                   width: '150%', 
                   height: '150%', 
                   background: `radial-gradient(circle, ${getScoreColor(confidence_score)}15 0%, transparent 70%)`,
                   zIndex: -1 
                 }} />
              </div>
              <div className="badge" style={{ 
                backgroundColor: `${getScoreColor(confidence_score)}15`, 
                color: getScoreColor(confidence_score),
                borderColor: `${getScoreColor(confidence_score)}30`,
                padding: '8px 16px',
                borderRadius: '100px'
              }}>
                {confidence_score >= 80 ? 'EXCELLENT' : confidence_score >= 50 ? 'AVERAGE' : 'IMPROVING'}
              </div>
           </motion.section>

           {/* METRICS GRID */}
           <div className="metrics-grid">
              <motion.div variants={itemVariants} className="metric-card glass-panel">
                 <div className="metric-card-header">
                    <Zap size={20} style={{ color: '#fbbf24' }} />
                    <span className="metric-card-label">Pace</span>
                 </div>
                 <span className="metric-card-val">{metrics.speech_rate} <small style={{ fontSize: '14px', opacity: 0.5 }}>WPM</small></span>
              </motion.div>

              <motion.div variants={itemVariants} className="metric-card glass-panel">
                 <div className="metric-card-header">
                    <MessageSquare size={20} style={{ color: '#f87171' }} />
                    <span className="metric-card-label">Fillers</span>
                 </div>
                 <span className="metric-card-val">{metrics.filler_words} <small style={{ fontSize: '14px', opacity: 0.5 }}>Used</small></span>
              </motion.div>

              <motion.div variants={itemVariants} className="metric-card glass-panel">
                 <div className="metric-card-header">
                    <Clock size={20} style={{ color: '#60a5fa' }} />
                    <span className="metric-card-label">Length</span>
                 </div>
                 <span className="metric-card-val">{formatDuration(duration)}</span>
              </motion.div>

              <motion.div variants={itemVariants} className="metric-card glass-panel">
                 <div className="metric-card-header">
                    <div style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: getEmotionColor(emotion.label) }} />
                    <span className="metric-card-label">Emotion</span>
                 </div>
                 <span className="metric-card-val" style={{ color: getEmotionColor(emotion.label) }}>{emotion.label}</span>
              </motion.div>
           </div>

           {/* AI FEEDBACK PANEL */}
           <motion.section variants={itemVariants} className="ai-feedback-panel glass-panel">
              <div className="feedback-header">
                 <ShieldCheck className="success" size={24} />
                 <h3 style={{ margin: 0, fontSize: '20px' }}>AI Coaching Insights</h3>
              </div>
              
              <div className="feedback-columns">
                <div className="feedback-column">
                   <div className="column-title" style={{ color: '#10b981' }}>
                      <CheckCircle2 size={18} />
                      <h4>Strengths</h4>
                   </div>
                   <ul className="feedback-list">
                      {feedback.strengths.map((s, i) => (
                        <li key={i} style={{ color: '#10b981' }}>{s}</li>
                      ))}
                   </ul>
                </div>

                <div className="feedback-column">
                   <div className="column-title" style={{ color: '#f59e0b' }}>
                      <AlertCircle size={18} />
                      <h4>To Improve</h4>
                   </div>
                   <ul className="feedback-list">
                      {feedback.weaknesses.map((w, i) => (
                        <li key={i} style={{ color: '#f59e0b' }}>{w}</li>
                      ))}
                   </ul>
                </div>

                <div className="feedback-column">
                   <div className="column-title" style={{ color: '#6366f1' }}>
                      <Lightbulb size={18} />
                      <h4>Strategy</h4>
                   </div>
                   <ul className="feedback-list">
                      {feedback.suggestions.map((p, i) => (
                        <li key={i} style={{ color: '#818cf8' }}>{p}</li>
                      ))}
                   </ul>
                </div>
              </div>
           </motion.section>

           {/* TRANSCRIPT PANEL */}
           <motion.section variants={itemVariants} className="chart-panel glass-panel transcript-section">
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                 <LayoutDashboard size={20} className="text-secondary" />
                 <h3 style={{ margin: 0 }}>Full Transcription</h3>
              </div>
              <div className="transcript-text">
                {transcription || "No clear speech detected in this recording."}
              </div>
           </motion.section>

        </div>

        <motion.div 
          variants={itemVariants} 
          style={{ display: 'flex', justifyContent: 'center', marginTop: '40px' }}
        >
           <Link to="/practice" className="primary-hud-btn start-btn" style={{ textDecoration: 'none', padding: '16px 48px', borderRadius: '100px' }}>
              <span>PRACTICE AGAIN</span>
           </Link>
        </motion.div>

      </motion.div>
    </DashboardLayout>
  );
}
