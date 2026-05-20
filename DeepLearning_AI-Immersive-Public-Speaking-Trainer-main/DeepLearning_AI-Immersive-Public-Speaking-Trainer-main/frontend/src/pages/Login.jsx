import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock } from 'lucide-react';
import AuthLayout from '../components/AuthLayout';
import Input from '../components/Input';
import Button from '../components/Button';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';
import './AuthForms.css';

export default function Login() {
  const [formData, setFormData] = useState({ email: '', password: '', rememberMe: false });
  const [errors, setErrors] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();
  const { addToast } = useToast();
  const navigate = useNavigate();

  const validateForm = () => {
    const newErrors = {};
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    try {
      setIsLoading(true);
      console.log(`Starting login attempt for: ${formData.email}`);
      
      const result = await login(formData.email, formData.password);
      console.log("Login result:", result);

      if (result === true) {
        console.log("Navigating to home...");
        navigate("/", { replace: true });
        
        setTimeout(() => {
          navigate("/");
        }, 500);
      }
    } catch (error) {
      console.error("Login error:", error);
      addToast(error.message || 'An unexpected error occurred.', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleLogin = () => {
    addToast('Google Login is coming soon!', 'info');
  };

  return (
    <AuthLayout 
      title="Welcome Back" 
      subtitle="Train your confidence with AI"
    >
      <form onSubmit={handleSubmit} className="auth-form">
        <Input
          label="Email address"
          type="email"
          placeholder="name@example.com"
          icon={Mail}
          value={formData.email}
          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
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
            onChange={(e) => setFormData({ ...formData, password: e.target.value })}
            error={errors.password}
            disabled={isLoading}
          />
        </div>

        <div className="form-options">
          <label className="checkbox-container">
            <input 
              type="checkbox" 
              checked={formData.rememberMe}
              onChange={(e) => setFormData({ ...formData, rememberMe: e.target.checked })}
              disabled={isLoading}
            />
            <span className="checkmark glass-panel"></span>
            <span className="checkbox-label">Remember me</span>
          </label>
          <Link to="/forgot-password" className="forgot-password-link">Forgot password?</Link>
        </div>

        <Button type="submit" isLoading={isLoading} className="mt-4">
          Sign In
        </Button>

        <div className="divider">
          <span>or continue with</span>
        </div>

        <Button 
          type="button" 
          variant="secondary" 
          onClick={handleGoogleLogin}
          disabled={isLoading}
          icon={() => (
            <svg viewBox="0 0 24 24" width="20" height="20" xmlns="http://www.w3.org/2000/svg">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.47 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
          )}
        >
          Google
        </Button>
      </form>

      <p className="auth-footer">
        Don't have an account? <Link to="/signup" className="auth-footer-link">Sign up</Link>
      </p>
    </AuthLayout>
  );
}
