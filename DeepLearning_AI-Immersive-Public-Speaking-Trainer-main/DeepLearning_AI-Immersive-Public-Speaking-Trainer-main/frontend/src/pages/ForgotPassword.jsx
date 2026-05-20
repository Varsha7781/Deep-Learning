import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Mail, ArrowLeft, CheckCircle2, ShieldCheck } from 'lucide-react';
import AuthLayout from '../components/AuthLayout';
import Input from '../components/Input';
import Button from '../components/Button';
import './AuthForms.css';
import './ForgotPassword.css';

export default function ForgotPassword() {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSent, setIsSent] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email) {
      setError('Email is required');
      return;
    } else if (!/\S+@\S+\.\S+/.test(email)) {
      setError('Please enter a valid email address');
      return;
    }
    setError('');
    setIsLoading(true);

    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      setIsSent(true);
    }, 1500);
  };

  if (isSent) {
    return (
      <AuthLayout
        title="Check your inbox"
        subtitle={`We've sent a password reset link to`}
      >
        <div className="forgot-success">
          {/* Animated success icon */}
          <div className="forgot-success-icon-wrapper">
            <div className="forgot-success-ring"></div>
            <div className="forgot-success-icon">
              <CheckCircle2 size={40} strokeWidth={1.5} />
            </div>
          </div>

          {/* Email display */}
          <div className="forgot-success-email">
            <Mail size={16} />
            <span>{email}</span>
          </div>

          {/* Info message */}
          <p className="forgot-success-info">
            Click the link in the email to create a new password. 
            If you don't see it, check your spam folder.
          </p>

          {/* Actions */}
          <div className="forgot-success-actions">
            <Button variant="primary" onClick={() => window.location.href = '/login'}>
              <ArrowLeft size={18} />
              Back to login
            </Button>

            <button
              className="forgot-resend-btn"
              onClick={() => {
                setIsSent(false);
                setEmail(email);
              }}
            >
              Didn't receive it? <span className="forgot-resend-link">Resend email</span>
            </button>
          </div>
        </div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      title="Forgot password?"
      subtitle="Don't worry, we've got you covered"
    >
      {/* Reassurance badge */}
      <div className="forgot-reassurance">
        <ShieldCheck size={16} />
        <span>We'll send a secure reset link to your email</span>
      </div>

      <form onSubmit={handleSubmit} className="auth-form" id="forgot-password-form">
        <Input
          id="forgot-email-input"
          label="Email address"
          type="email"
          placeholder="name@example.com"
          icon={Mail}
          value={email}
          onChange={(e) => {
            setEmail(e.target.value);
            if (error) setError('');
          }}
          error={error}
        />

        <Button type="submit" isLoading={isLoading} className="mt-4" id="forgot-submit-btn">
          Send Reset Link
        </Button>
      </form>

      {/* Back to login */}
      <div className="forgot-back-row">
        <Link to="/login" className="forgot-back-link" id="forgot-back-to-login">
          <ArrowLeft size={16} />
          Back to login
        </Link>
      </div>
    </AuthLayout>
  );
}
