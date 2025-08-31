# Son of Andrew - Frontend

A modern, responsive chat interface for the Son of Andrew conversational agent system, built with Next.js 15, React 19, and Tailwind CSS.

## Features

- **💬 Real-time Chat Interface** - Seamless conversation with Andrew's AI system
- **📱 Responsive Design** - Works perfectly on desktop, tablet, and mobile
- **💾 Session Persistence** - Conversations saved locally with cache validation
- **⚡ Performance Optimized** - Fast loading with intelligent caching
- **🎨 Modern UI** - Clean, accessible interface with smooth animations
- **⏱️ Response Timing** - Real-time response time tracking
- **🔄 Auto-resize Input** - Text area adapts to content length
- **🧠 Memory Integration** - Seamless integration with backend memory system

## Tech Stack

- **Next.js 15** - React framework with App Router
- **React 19** - Latest React with concurrent features
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Lucide React** - Modern icon library

## Getting Started

### Prerequisites

- Node.js 18+ with npm
- Backend server running on port 8000

### 🚀 Quick Start

#### 1. Install Dependencies

```bash
# Navigate to frontend directory (use absolute path)
cd /Users/andreweaton/son-of-andrew_frontend

# Install all dependencies
npm install
```

#### 2. Start Development Server

```bash
# Start the development server (from frontend directory)
cd /Users/andreweaton/son-of-andrew_frontend
npm run dev
```

#### 3. Access the Application

- **Frontend**: http://localhost:3000
- **Network**: http://192.168.0.4:3000 (or your local IP)

### 🔧 Development Workflow

#### Complete Startup Process

**⚠️ CRITICAL STARTUP COMMANDS** (use these EXACT commands - no variations):

**Step 1: Start Backend (Terminal 1)**
```bash
# BACKEND LOCATION: /Users/andreweaton/son-of-andrew/
cd /Users/andreweaton/son-of-andrew
source venv/bin/activate
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

**Step 2: Start Frontend (Terminal 2)**
```bash
# FRONTEND LOCATION: /Users/andreweaton/son-of-andrew_frontend/
cd /Users/andreweaton/son-of-andrew_frontend
npm run dev
```

**🚨 IMPORTANT NOTES:**
- Frontend is in `son-of-andrew_frontend` directory (NOT inside the backend directory)
- Backend is in `son-of-andrew` directory 
- Both directories are at the same level in `/Users/andreweaton/`
- Use ABSOLUTE paths if relative paths fail

#### Verify Everything is Working

1. **Backend Health Check**: http://localhost:8000/health
2. **Frontend Interface**: http://localhost:3000
3. **Test Chat**: Send a message through the interface

### 📁 Project Structure

```
son-of-andrew_frontend/
├── src/
│   └── app/
│       ├── layout.tsx      # Root layout with global styles
│       ├── page.tsx        # Main chat interface
│       └── globals.css     # Global CSS with Tailwind
├── public/
│   └── favicon.ico         # Site favicon
├── package.json            # Dependencies and scripts
├── tailwind.config.js      # Tailwind CSS configuration
├── next.config.ts          # Next.js configuration
└── tsconfig.json          # TypeScript configuration
```

### 🎨 Tailwind CSS Configuration

The Tailwind config is set up to work with the `src/app/` directory structure:

```javascript
// tailwind.config.js
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  // ... rest of config
};
```

### 🔧 Available Scripts

```bash
# Development server
npm run dev

# Production build
npm run build

# Start production server
npm start

# Type checking
npm run type-check

# Linting
npm run lint
```

### 🐛 Common Issues & Solutions

#### Problem: "No utility classes were detected"
**Solution**: Restart the development server after any Tailwind config changes:
```bash
# Stop the server (Ctrl+C)
npm run dev
```

#### Problem: "Cannot connect to backend"
**Solution**: Ensure the backend is running on port 8000:
```bash
# In backend directory
cd son-of-andrew
source venv/bin/activate
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

#### Problem: "Module not found" errors
**Solution**: Reinstall dependencies:
```bash
rm -rf node_modules package-lock.json
npm install
```

#### Problem: Port 3000 already in use
**Solution**: Next.js will automatically use the next available port (3001, 3002, etc.)

### 🌐 API Integration

The frontend communicates with the backend through these endpoints:

- **Chat**: `POST http://localhost:8000/api/chat`
- **Health**: `GET http://localhost:8000/health`

### 💾 Local Storage

The application uses browser localStorage for:
- Conversation history
- Session management
- Cache validation
- User preferences

### 🎯 Features in Detail

#### Chat Interface
- Auto-expanding textarea
- Send button with loading states
- Keyboard shortcuts (Enter to send, Shift+Enter for new line)
- Message history with timestamps

#### Session Management
- Automatic session creation
- Conversation persistence
- Cache validation with backend
- Session restoration on reload

#### Performance
- Optimized re-renders
- Efficient state management
- Lazy loading where appropriate
- Minimal bundle size

### 🔄 Development Tips

1. **Hot Reload**: Changes to code automatically refresh the browser
2. **Type Safety**: Use TypeScript for better development experience
3. **CSS Classes**: Use Tailwind utilities for consistent styling
4. **Component Structure**: Keep components focused and reusable

### 🧪 Testing

```bash
# Test the chat interface
# 1. Send a message
# 2. Verify response appears
# 3. Check session persistence
# 4. Test keyboard shortcuts

# Test API connectivity
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello from frontend test"}'
```

### 🚀 Production Deployment

#### Build for Production

```bash
# Create optimized build
npm run build

# Start production server
npm start
```

#### Environment Variables

Create `.env.local` for production settings:

```env
# API Configuration
NEXT_PUBLIC_API_URL=https://your-backend-domain.com

# Analytics (optional)
NEXT_PUBLIC_GA_ID=your-google-analytics-id
```

### 🔧 Customization

#### Styling
- Modify `src/app/globals.css` for global styles
- Update `tailwind.config.js` for theme customization
- Use Tailwind utilities in components

#### Features
- Add new chat features in `src/app/page.tsx`
- Extend API integration as needed
- Customize localStorage behavior

### 🤝 Contributing

1. Follow the existing code structure
2. Use TypeScript for type safety
3. Follow Tailwind CSS patterns
4. Test on multiple screen sizes
5. Ensure accessibility compliance

### 📚 Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [React 19 Features](https://react.dev/blog/2024/04/25/react-19)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [TypeScript](https://www.typescriptlang.org/docs)

## Troubleshooting

### Build Issues

1. **TypeScript Errors**: Run `npm run type-check` to identify issues
2. **Tailwind Not Working**: Verify config paths match your file structure
3. **Import Errors**: Check file paths and exports

### Runtime Issues

1. **API Connection**: Verify backend is running and accessible
2. **CORS Errors**: Check backend CORS configuration
3. **Session Issues**: Clear localStorage and restart

### Performance Issues

1. **Slow Loading**: Check network tab for large resources
2. **Memory Leaks**: Monitor browser dev tools
3. **Bundle Size**: Analyze with `npm run build`

For backend-related issues, see the main [README](../son-of-andrew/README.md).
