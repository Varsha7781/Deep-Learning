import React from 'react';
import { Mic2 } from 'lucide-react';
import './AuthLayout.css';

export default function AuthLayout({ children, title, subtitle }) {
  return (
    <div className="auth-layout bg-animated-gradient">
      <div className="auth-container">
        
        {/* Core Glassmorphism Card */}
        <div className="auth-card glass-panel animate-slide-up">
          
          <div className="auth-header text-center">
            <div className="auth-logo-container animate-float">
              <div className="auth-logo-bg">
                <Mic2 size={32} color="#ffffff" />
              </div>
            </div>
            {title && <h1 className="auth-title animate-fade-in delay-100">{title}</h1>}
            {subtitle && <p className="auth-subtitle animate-fade-in delay-200">{subtitle}</p>}
          </div>

          <div className="auth-content animate-fade-in delay-300">
            {children}
          </div>

        </div>
      </div>
    </div>
  );
}
