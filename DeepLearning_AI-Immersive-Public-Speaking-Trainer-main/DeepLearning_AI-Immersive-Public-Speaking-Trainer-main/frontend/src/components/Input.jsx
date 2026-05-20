import React, { useState } from 'react';
import { Eye, EyeOff } from 'lucide-react';
import './Input.css';

export default function Input({ label, error, type = 'text', icon: Icon, ...props }) {
  const [showPassword, setShowPassword] = useState(false);
  const isPassword = type === 'password';
  const inputType = isPassword && showPassword ? 'text' : type;

  return (
    <div className="input-container">
      {label && <label className="input-label">{label}</label>}
      <div className="input-wrapper">
        {Icon && <Icon className="input-icon-left" size={20} />}
        <input 
          type={inputType} 
          className={`input-field ${error ? 'input-error' : ''} ${Icon ? 'has-icon-left' : ''} ${isPassword ? 'has-icon-right' : ''}`}
          {...props}
        />
        {isPassword && (
          <button 
            type="button" 
            className="input-icon-right" 
            onClick={() => setShowPassword(!showPassword)}
            tabIndex="-1"
          >
            {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
          </button>
        )}
      </div>
      {error && <span className="input-error-message animate-fade-in">{error}</span>}
    </div>
  );
}
