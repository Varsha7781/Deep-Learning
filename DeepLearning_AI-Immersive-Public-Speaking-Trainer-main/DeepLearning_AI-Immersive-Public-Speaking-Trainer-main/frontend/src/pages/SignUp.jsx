import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock, User, ShieldCheck } from 'lucide-react';
import AuthLayout from '../components/AuthLayout';
import Input from '../components/Input';
import Button from '../components/Button';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';
import './AuthForms.css';

const strengthLabels = ['', 'Weak', 'Fair', 'Strong'];
const strengthColors = ['', 'strength-1', 'strength-2', 'strength-3'];

export default function SignUp() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    agreeTerms: false,
  });
  const [errors, setErrors] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState(0);
  const { register } = useAuth();
  const { addToast } = useToast();
  const navigate = useNavigate();

  const calculateStrength = (pass) => {
    let score = 0;
    if (!pass) return 0;
    if (pass.length >= 8) score += 1;
    if (/[A-Z]/.test(pass) && /[a-z]/.test(pass)) score += 1;
    if (/[0-9]/.test(pass) && /[^A-Za-z0-9]/.test(pass)) score += 1;
    return score;
  };

  const getStrengthHint = () => {
    const hints = [];
    if (formData.password.length < 8) hints.push('8+ characters');
    if (!/[A-Z]/.test(formData.password) || !/[a-z]/.test(formData.password)) hints.push('upper & lowercase');
    if (!/[0-9]/.test(formData.password)) hints.push('a number');
    if (!/[^A-Za-z0-9]/.test(formData.password)) hints.push('a special character');
    return hints.length > 0 ? `Add ${hints.join(', ')}` : '';
  };

  const handlePasswordChange = (e) => {
    const val = e.target.value;
    setFormData({ ...formData, password: val });
    setPasswordStrength(calculateStrength(val));
    if (errors.password) setErrors({ ...errors, password: '' });
  };

  const clearFieldError = (field) => {
    if (errors[field]) {
      const newErrors = { ...errors };
      delete newErrors[field];
      setErrors(newErrors);
    }
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.name.trim()) {
      newErrors.name = 'Full name is required';
    }

    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (passwordStrength < 2) {
      newErrors.password = 'Password is too weak — see suggestions below';
    }

    if (!formData.confirmPassword) {
      newErrors.confirmPassword = 'Please confirm your password';
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }

    if (!formData.agreeTerms) {
      newErrors.agreeTerms = 'You must agree to the Terms & Conditions';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    try {
      setIsLoading(true);
      console.log(`Starting registration for: ${formData.email}`);
      
      const result = await register(formData.name, formData.email, formData.password);

      console.log("Signup result:", result);

      if (result) {
        navigate("/");
      }
    } catch (error) {
      console.error("Sign up error:", error);
      addToast(error.message || 'An unexpected error occurred.', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <AuthLayout
      title="Create an account"
      subtitle="Start your public speaking journey"
    >
      <form onSubmit={handleSubmit} className="auth-form">
        <Input
          label="Full Name"
          type="text"
          placeholder="Jane Doe"
          icon={User}
          value={formData.name}
          onChange={(e) => {
            setFormData({ ...formData, name: e.target.value });
            clearFieldError('name');
          }}
          error={errors.name}
          disabled={isLoading}
        />

        <Input
          label="Email address"
          type="email"
          placeholder="name@example.com"
          icon={Mail}
          value={formData.email}
          onChange={(e) => {
            setFormData({ ...formData, email: e.target.value });
            clearFieldError('email');
          }}
          error={errors.email}
          disabled={isLoading}
        />

        <div className="password-field">
          <Input
            label="Password"
            type="password"
            placeholder="••••••••"
            icon={Lock}
            value={formData.password}
            onChange={handlePasswordChange}
            error={errors.password}
            disabled={isLoading}
          />
          {formData.password && (
            <>
              <div className="password-strength animate-fade-in">
                {[1, 2, 3].map((level) => (
                  <div
                    key={level}
                    className={`strength-bar ${passwordStrength >= level ? strengthColors[level] : ''}`}
                  />
                ))}
              </div>
              <div className="strength-info">
                <span className={`strength-label ${strengthColors[passwordStrength]}-text`}>
                  {strengthLabels[passwordStrength]}
                </span>
                {passwordStrength < 3 && (
                  <span className="strength-hint">{getStrengthHint()}</span>
                )}
              </div>
            </>
          )}
        </div>

        <Input
          label="Confirm Password"
          type="password"
          placeholder="••••••••"
          icon={ShieldCheck}
          value={formData.confirmPassword}
          onChange={(e) => {
            setFormData({ ...formData, confirmPassword: e.target.value });
            clearFieldError('confirmPassword');
          }}
          error={errors.confirmPassword}
          disabled={isLoading}
        />

        <div className="terms-container">
          <label className="checkbox-container">
            <input
              type="checkbox"
              checked={formData.agreeTerms}
              onChange={(e) => {
                setFormData({ ...formData, agreeTerms: e.target.checked });
                clearFieldError('agreeTerms');
              }}
              disabled={isLoading}
            />
            <span className="checkmark"></span>
            <span className="checkbox-label">
              I agree to the{' '}
              <a href="#" className="terms-link" onClick={(e) => e.preventDefault()}>
                Terms of Service
              </a>{' '}
              and{' '}
              <a href="#" className="terms-link" onClick={(e) => e.preventDefault()}>
                Privacy Policy
              </a>
            </span>
          </label>
          {errors.agreeTerms && (
            <span className="input-error-message terms-error animate-fade-in">{errors.agreeTerms}</span>
          )}
        </div>

        <Button type="submit" isLoading={isLoading} className="mt-4">
          Create Account
        </Button>
      </form>

      <p className="auth-footer">
        Already have an account?{' '}
        <Link to="/login" className="auth-footer-link">Sign in</Link>
      </p>
    </AuthLayout>
  );
}
