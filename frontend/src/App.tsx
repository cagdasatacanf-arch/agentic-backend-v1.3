import React, { useState, useEffect, useRef } from 'react';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || '';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  tools?: string[];
  sources?: Source[];
  session_id?: string;
}

interface Source {
  content: string;
  score: number;
  metadata?: {
    filename?: string;
    [key: string]: any;
  };
}

interface HealthStatus {
  api: 'online' | 'degraded' | 'offline';
  qdrant: 'online' | 'degraded' | 'offline';
  redis: 'online' | 'degraded' | 'offline';
}

const AgenticConsole = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTools, setActiveTools] = useState<string[]>([]);
  const [systemStatus, setSystemStatus] = useState<HealthStatus>({
    api: 'online',
    qdrant: 'online',
    redis: 'online'
  });
  const [showSources, setShowSources] = useState(false);
  const [currentSources, setCurrentSources] = useState<Source[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const tools = [
    { id: 'calculator', name: 'CALC', icon: '∑' },
    { id: 'web_search', name: 'WEB', icon: '◎' },
    { id: 'search_documents', name: 'RAG', icon: '⬡' },
    { id: 'read_file', name: 'FILE', icon: '▤' },
    { id: 'run_python', name: 'CODE', icon: '⌘' },
    { id: 'make_http_request', name: 'HTTP', icon: '⟲' },
  ];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    // Initialize session and check health
    const init = async () => {
      setConversationId(`session_${Date.now()}`);
      await checkHealth();
      await fetchMetadata();
    };
    init();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/health`);
      if (response.ok) {
        const data = await response.json();
        const newStatus: HealthStatus = {
          api: data.status === 'healthy' ? 'online' : 'degraded',
          qdrant: data.services?.qdrant === 'healthy' ? 'online' : 'degraded',
          redis: data.services?.redis === 'healthy' ? 'online' : 'degraded',
        };
        setSystemStatus(newStatus);
      }
    } catch (error) {
      console.error('Health check failed:', error);
      setSystemStatus({ api: 'offline', qdrant: 'offline', redis: 'offline' });
    }
  };

  const fetchMetadata = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/metadata`);
      if (response.ok) {
        const data = await response.json();
        setMetadata(data);
      }
    } catch (error) {
      console.error('Metadata fetch failed:', error);
    }
  };

  const detectTools = (question: string): string[] => {
    const detected: string[] = [];
    const lower = question.toLowerCase();

    if (lower.match(/\d+[\+\-\*\/]\d+/) || lower.includes('calculate')) {
      detected.push('calculator');
    }
    if (lower.includes('search') || lower.includes('find') || lower.includes('look up')) {
      detected.push('web_search');
    }
    if (lower.includes('document') || lower.includes('file') || lower.includes('knowledge')) {
      detected.push('search_documents');
    }
    if (lower.includes('read') && lower.includes('file')) {
      detected.push('read_file');
    }
    if (lower.includes('code') || lower.includes('python') || lower.includes('run')) {
      detected.push('run_python');
    }

    return detected.length > 0 ? detected : ['search_documents'];
  };

  const simulateToolUsage = (toolIds: string[]) => {
    setActiveTools(toolIds);
    setTimeout(() => setActiveTools([]), 2000);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setIsProcessing(true);

    // Detect and show potential tools
    const detectedTools = detectTools(currentInput);
    simulateToolUsage(detectedTools);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/langgraph/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY,
        },
        body: JSON.stringify({
          question: currentInput,
          use_rag: detectedTools.includes('search_documents'),
          session_id: conversationId,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      // Extract tools used from metadata
      const toolsUsed = data.metadata?.tools_used || detectedTools;

      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        tools: toolsUsed,
        sources: data.sources || [],
        session_id: data.session_id,
      };

      setMessages(prev => [...prev, assistantMessage]);

      if (data.sources && data.sources.length > 0) {
        setCurrentSources(data.sources);
      }

      // Update conversation ID if new one was created
      if (data.session_id && data.session_id !== conversationId) {
        setConversationId(data.session_id);
      }

    } catch (error) {
      console.error('Query failed:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to process request'}. Please check that the backend is running and API key is configured.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const StatusDot = ({ status }: { status: 'online' | 'degraded' | 'offline' }) => (
    <span style={{
      display: 'inline-block',
      width: '6px',
      height: '6px',
      borderRadius: '50%',
      backgroundColor: status === 'online' ? '#00ff88' : status === 'degraded' ? '#ffaa00' : '#ff4444',
      marginRight: '6px',
      animation: status === 'online' ? 'pulse 2s infinite' : 'none',
    }} />
  );

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#0a0a0a',
      color: '#e8e8e8',
      fontFamily: "'IBM Plex Mono', 'JetBrains Mono', monospace",
      display: 'flex',
      flexDirection: 'column',
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        @keyframes toolFlash {
          0%, 100% { background-color: rgba(0, 255, 136, 0.1); }
          50% { background-color: rgba(0, 255, 136, 0.3); }
        }

        @keyframes slideIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes typing {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }

        @keyframes scanline {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }

        * {
          box-sizing: border-box;
        }

        ::-webkit-scrollbar {
          width: 4px;
        }

        ::-webkit-scrollbar-track {
          background: #1a1a1a;
        }

        ::-webkit-scrollbar-thumb {
          background: #333;
          border-radius: 2px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: #444;
        }

        input::placeholder {
          color: #555;
        }
      `}</style>

      {/* Scanline Effect */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: '2px',
        background: 'linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.1), transparent)',
        animation: 'scanline 8s linear infinite',
        pointerEvents: 'none',
        zIndex: 1000,
      }} />

      {/* Header */}
      <header style={{
        borderBottom: '1px solid #222',
        padding: '16px 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'linear-gradient(180deg, #111 0%, #0a0a0a 100%)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            width: '36px',
            height: '36px',
            border: '2px solid #00ff88',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '16px',
            fontWeight: '600',
            color: '#00ff88',
          }}>
            ⬢
          </div>
          <div>
            <h1 style={{
              margin: 0,
              fontSize: '14px',
              fontWeight: '600',
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              fontFamily: "'Space Grotesk', sans-serif",
            }}>
              AGENTIC CONSOLE
            </h1>
            <p style={{
              margin: '2px 0 0',
              fontSize: '10px',
              color: '#666',
              letterSpacing: '0.05em',
            }}>
              v1.3 • {metadata?.chat_model?.toUpperCase() || 'LOADING...'}
            </p>
          </div>
        </div>

        {/* System Status */}
        <div style={{
          display: 'flex',
          gap: '20px',
          fontSize: '10px',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <StatusDot status={systemStatus.api} />
            <span style={{ color: '#888' }}>API</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <StatusDot status={systemStatus.qdrant} />
            <span style={{ color: '#888' }}>QDRANT</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <StatusDot status={systemStatus.redis} />
            <span style={{ color: '#888' }}>REDIS</span>
          </div>
        </div>
      </header>

      {/* Tool Bar */}
      <div style={{
        padding: '12px 24px',
        borderBottom: '1px solid #1a1a1a',
        display: 'flex',
        gap: '8px',
        overflowX: 'auto',
      }}>
        {tools.map(tool => (
          <div
            key={tool.id}
            style={{
              padding: '6px 12px',
              border: `1px solid ${activeTools.includes(tool.id) ? '#00ff88' : '#333'}`,
              borderRadius: '2px',
              fontSize: '10px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              color: activeTools.includes(tool.id) ? '#00ff88' : '#666',
              backgroundColor: activeTools.includes(tool.id) ? 'rgba(0, 255, 136, 0.1)' : 'transparent',
              transition: 'all 0.2s ease',
              animation: activeTools.includes(tool.id) ? 'toolFlash 0.5s ease infinite' : 'none',
            }}
          >
            <span style={{ fontSize: '12px' }}>{tool.icon}</span>
            <span style={{ letterSpacing: '0.05em' }}>{tool.name}</span>
          </div>
        ))}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '10px', color: '#444' }}>SESSION:</span>
          <span style={{ fontSize: '10px', color: '#00ff88', fontFamily: 'monospace' }}>
            {conversationId?.slice(0, 16)}
          </span>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Messages Area */}
        <main style={{
          flex: 1,
          overflow: 'auto',
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
        }}>
          {messages.length === 0 && (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#333',
              textAlign: 'center',
              padding: '40px',
            }}>
              <div style={{
                width: '80px',
                height: '80px',
                border: '1px solid #222',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '24px',
              }}>
                <span style={{ fontSize: '32px', color: '#222' }}>⬢</span>
              </div>
              <p style={{
                fontSize: '12px',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                maxWidth: '400px',
                lineHeight: '1.8',
              }}>
                MULTI-AGENT SYSTEM READY
              </p>
              <p style={{
                fontSize: '11px',
                color: '#444',
                marginTop: '16px',
                maxWidth: '500px',
                lineHeight: '1.6',
              }}>
                {metadata ? `${metadata.chat_model} • ${metadata.embedding_model}` : 'Initializing...'}
              </p>

              {/* Quick Commands */}
              <div style={{
                marginTop: '40px',
                display: 'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap: '8px',
                width: '100%',
                maxWidth: '500px',
              }}>
                {[
                  'What is 25 * 47?',
                  'Search for latest AI news',
                  'Analyze my documents',
                  'Explain quantum computing',
                ].map((cmd, i) => (
                  <button
                    key={i}
                    onClick={() => setInput(cmd)}
                    style={{
                      padding: '12px 16px',
                      border: '1px solid #222',
                      background: 'transparent',
                      color: '#555',
                      fontSize: '11px',
                      textAlign: 'left',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      fontFamily: 'inherit',
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.borderColor = '#00ff88';
                      e.currentTarget.style.color = '#00ff88';
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.borderColor = '#222';
                      e.currentTarget.style.color = '#555';
                    }}
                  >
                    → {cmd}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              style={{
                marginBottom: '24px',
                animation: 'slideIn 0.3s ease',
              }}
            >
              {/* Message Header */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                marginBottom: '8px',
              }}>
                <div style={{
                  width: '24px',
                  height: '24px',
                  border: `1px solid ${msg.role === 'user' ? '#666' : '#00ff88'}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '10px',
                  color: msg.role === 'user' ? '#666' : '#00ff88',
                }}>
                  {msg.role === 'user' ? '>' : '⬢'}
                </div>
                <span style={{
                  fontSize: '10px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.1em',
                  color: msg.role === 'user' ? '#666' : '#00ff88',
                }}>
                  {msg.role === 'user' ? 'INPUT' : 'AGENT'}
                </span>
                <span style={{
                  fontSize: '10px',
                  color: '#333',
                  marginLeft: 'auto',
                }}>
                  {msg.timestamp.toLocaleTimeString('en-US', {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false
                  })}
                </span>
              </div>

              {/* Message Content */}
              <div style={{
                paddingLeft: '36px',
                fontSize: '13px',
                lineHeight: '1.7',
                color: msg.role === 'user' ? '#aaa' : '#e8e8e8',
                whiteSpace: 'pre-wrap',
              }}>
                {msg.content}
              </div>

              {/* Tools Used */}
              {msg.tools && msg.tools.length > 0 && (
                <div style={{
                  paddingLeft: '36px',
                  marginTop: '12px',
                  display: 'flex',
                  gap: '6px',
                  flexWrap: 'wrap',
                }}>
                  {msg.tools.map((toolId, i) => {
                    const tool = tools.find(t => t.id === toolId);
                    return tool ? (
                      <span
                        key={i}
                        style={{
                          padding: '3px 8px',
                          border: '1px solid #333',
                          fontSize: '9px',
                          color: '#666',
                          letterSpacing: '0.05em',
                        }}
                      >
                        {tool.icon} {tool.name}
                      </span>
                    ) : null;
                  })}
                </div>
              )}

              {/* Sources Link */}
              {msg.sources && msg.sources.length > 0 && (
                <button
                  onClick={() => {
                    setCurrentSources(msg.sources || []);
                    setShowSources(true);
                  }}
                  style={{
                    marginLeft: '36px',
                    marginTop: '12px',
                    padding: '6px 12px',
                    border: '1px solid #333',
                    background: 'transparent',
                    color: '#666',
                    fontSize: '10px',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                    letterSpacing: '0.05em',
                    transition: 'all 0.2s ease',
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.borderColor = '#00ff88';
                    e.currentTarget.style.color = '#00ff88';
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.borderColor = '#333';
                    e.currentTarget.style.color = '#666';
                  }}
                >
                  VIEW {msg.sources.length} SOURCES →
                </button>
              )}
            </div>
          ))}

          {/* Processing Indicator */}
          {isProcessing && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              padding: '16px 0',
              animation: 'slideIn 0.3s ease',
            }}>
              <div style={{
                width: '24px',
                height: '24px',
                border: '1px solid #00ff88',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '10px',
                color: '#00ff88',
              }}>
                ⬢
              </div>
              <span style={{
                fontSize: '12px',
                color: '#00ff88',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}>
                Processing
                <span style={{ animation: 'typing 1s infinite' }}>▊</span>
              </span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </main>

        {/* Sources Panel */}
        {showSources && (
          <aside style={{
            width: '320px',
            borderLeft: '1px solid #1a1a1a',
            overflow: 'auto',
            animation: 'slideIn 0.3s ease',
          }}>
            <div style={{
              padding: '16px',
              borderBottom: '1px solid #1a1a1a',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <span style={{
                fontSize: '10px',
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
                color: '#666',
              }}>
                RAG SOURCES
              </span>
              <button
                onClick={() => setShowSources(false)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#666',
                  cursor: 'pointer',
                  fontSize: '16px',
                  padding: '4px',
                }}
              >
                ×
              </button>
            </div>
            <div style={{ padding: '16px' }}>
              {currentSources.map((source, i) => (
                <div
                  key={i}
                  style={{
                    padding: '12px',
                    border: '1px solid #222',
                    marginBottom: '12px',
                  }}
                >
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '8px',
                  }}>
                    <span style={{
                      fontSize: '10px',
                      color: '#00ff88',
                      letterSpacing: '0.05em',
                    }}>
                      {source.metadata?.filename || `Source #${i + 1}`}
                    </span>
                    <span style={{
                      fontSize: '10px',
                      color: '#444',
                      padding: '2px 6px',
                      border: '1px solid #333',
                    }}>
                      {(source.score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p style={{
                    fontSize: '11px',
                    color: '#666',
                    lineHeight: '1.5',
                    margin: 0,
                  }}>
                    {source.content.slice(0, 200)}...
                  </p>
                </div>
              ))}
            </div>
          </aside>
        )}
      </div>

      {/* Input Area */}
      <form
        onSubmit={handleSubmit}
        style={{
          padding: '16px 24px',
          borderTop: '1px solid #1a1a1a',
          display: 'flex',
          gap: '12px',
          background: '#0d0d0d',
        }}
      >
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          border: '1px solid #222',
          padding: '0 16px',
          transition: 'border-color 0.2s ease',
        }}>
          <span style={{ color: '#00ff88', marginRight: '12px', fontSize: '14px' }}>›</span>
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter command or query..."
            disabled={isProcessing}
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              color: '#e8e8e8',
              fontSize: '13px',
              padding: '14px 0',
              outline: 'none',
              fontFamily: 'inherit',
            }}
          />
        </div>
        <button
          type="submit"
          disabled={isProcessing || !input.trim()}
          style={{
            padding: '0 24px',
            background: isProcessing || !input.trim() ? '#1a1a1a' : '#00ff88',
            border: 'none',
            color: isProcessing || !input.trim() ? '#444' : '#0a0a0a',
            fontSize: '11px',
            fontWeight: '600',
            letterSpacing: '0.1em',
            cursor: isProcessing || !input.trim() ? 'not-allowed' : 'pointer',
            fontFamily: "'Space Grotesk', sans-serif",
            transition: 'all 0.2s ease',
          }}
        >
          {isProcessing ? 'PROCESSING...' : 'EXECUTE'}
        </button>
      </form>

      {/* Footer */}
      <footer style={{
        padding: '8px 24px',
        borderTop: '1px solid #111',
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '9px',
        color: '#333',
        letterSpacing: '0.05em',
        textTransform: 'uppercase',
      }}>
        <span>FASTAPI • QDRANT • REDIS • OPENAI</span>
        <span>{systemStatus.api === 'online' ? 'SYSTEM OPERATIONAL' : 'SYSTEM DEGRADED'}</span>
      </footer>
    </div>
  );
};

export default AgenticConsole;
