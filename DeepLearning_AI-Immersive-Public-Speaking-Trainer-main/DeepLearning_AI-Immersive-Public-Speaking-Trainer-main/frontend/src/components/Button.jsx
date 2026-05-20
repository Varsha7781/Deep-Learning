import React from 'react';
import './Button.css';

export default function Button({ 
  children, 
  variant = 'primary', 
  isLoading = false, 
  icon: Icon,
  className = '',
  ...props 
}) {
  return (
    <button 
      className={`btn btn-${variant} ${className} ${isLoading ? 'loading' : ''}`}
      disabled={isLoading || props.disabled}
      {...props}
    >
      {isLoading ? (
        <svg className="btn-spinner" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      ) : Icon ? (
        <Icon className="btn-icon" size={20} />
      ) : null}
      <span className="btn-text">{children}</span>
    </button>
  );
}
