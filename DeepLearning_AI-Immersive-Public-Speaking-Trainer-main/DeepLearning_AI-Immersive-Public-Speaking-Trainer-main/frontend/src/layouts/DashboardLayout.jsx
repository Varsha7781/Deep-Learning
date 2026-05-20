import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { 
  Home, 
  Mic2, 
  BarChart2, 
  History, 
  Settings, 
  LogOut, 
  User,
  Plus
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';
import { motion } from 'framer-motion';
import './DashboardLayout.css';

export default function DashboardLayout({ children }) {
  const { user, logout } = useAuth();
  const { addToast } = useToast();
  const location = useLocation();
  const navigate = useNavigate();

  const menuItems = [
    { icon: Home, label: 'Home', path: '/' },
    { icon: Mic2, label: 'Practice', path: '/practice' },
    { icon: BarChart2, label: 'Analysis', path: '/analysis' },
    { icon: History, label: 'History', path: '/history' },
  ];

  const handleLogout = () => {
    addToast('Logging out...', 'info', 2000);
    logout();
  };

  return (
    <div className="dashboard-layout bg-animated-gradient">
      {/* Sidebar Navigation */}
      <aside className="sidebar glass-panel">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <Mic2 size={24} className="logo-icon" />
            <span className="logo-text">AI Coach</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {menuItems.map((item) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;
            return (
              <Link 
                key={item.path} 
                to={item.path} 
                className={`nav-item ${isActive ? 'active' : ''}`}
              >
                <Icon size={20} />
                <span>{item.label}</span>
                {isActive && (
                  <motion.div 
                    layoutId="active-indicator" 
                    className="active-indicator"
                  />
                )}
              </Link>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <div className="user-profile glass-panel">
            <div className="user-avatar">
              <User size={20} />
            </div>
            <div className="user-info">
              <span className="user-name">{user?.name || 'User'}</span>
              <span className="user-email">{user?.email || 'user@example.com'}</span>
            </div>
          </div>
          <button className="logout-btn" onClick={handleLogout}>
            <LogOut size={20} />
            <span>Logout</span>
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <header className="content-header">
           <div className="header-search">
              {/* Optional search or breadcrumbs */}
           </div>
           <div className="header-actions">
              <Link to="/practice" className="start-btn pulse-btn">
                 <Plus size={18} />
                 <span>New Session</span>
              </Link>
           </div>
        </header>
        <div className="content-inner">
          {children}
        </div>
      </main>
    </div>
  );
}
