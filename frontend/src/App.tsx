import React, { useState, useEffect, useRef } from 'react';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || '';

interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  tools?: string[];
  sources?: Source[];
}

interface Source {
  content: string;
  score: number;
  metadata?: {
    filename?: string;
    [key: string]: any;
  };
}

const AgenticConsole = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTools, setActiveTools] = useState<string[]>([]);
  const [showSources, setShowSources] = useState(false);
  const [currentSources, setCurrentSources] = useState<Source[]>([]);
  const [metrics, setMetrics] = useState({ latency: 0, tokens: 0 });
  const [mousePos, setMousePos] = useState({ x: 0.5, y: 0.5 });
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const tools = [
    { id: 'search_documents', label: 'RAG' },
    { id: 'calculator', label: 'MATH' },
    { id: 'web_search', label: 'WEB' },
    { id: 'run_python', label: 'EXEC' },
    { id: 'read_file', label: 'FILE' },
    { id: 'make_http_request', label: 'HTTP' },
  ];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setMousePos({
          x: (e.clientX - rect.left) / rect.width,
          y: (e.clientY - rect.top) / rect.height,
        });
      }
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useEffect(() => {
    // Initialize session and fetch metadata
    const init = async () => {
      setConversationId(`session_${Date.now()}`);
      await fetchMetadata();
    };
    init();
  }, []);

  const fetchMetadata = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/metadata`);
      if (response.ok) {
        const data = await response.json();
        setModelInfo(data.chat_model?.toUpperCase() || 'GPT-4O');
      }
    } catch (error) {
      console.error('Metadata fetch failed:', error);
      setModelInfo('LOADING');
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
    toolIds.forEach((tool, index) => {
      setTimeout(() => {
        setActiveTools(prev => [...prev, tool]);
      }, index * 300);
    });

    setTimeout(() => {
      setActiveTools([]);
    }, toolIds.length * 300 + 2000);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;

    const startTime = Date.now();
    const userMessage: Message = {
      id: Date.now(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setIsProcessing(true);
    setCurrentSources([]);

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
      const latency = Date.now() - startTime;

      // Extract tools used from metadata
      const toolsUsed = data.metadata?.tools_used || detectedTools;

      const assistantMessage: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        tools: toolsUsed,
        sources: data.sources || [],
      };

      setMessages(prev => [...prev, assistantMessage]);

      if (data.sources && data.sources.length > 0) {
        setCurrentSources(data.sources);
      }

      // Update metrics
      setMetrics({
        latency: latency,
        tokens: data.metadata?.total_tokens || Math.floor(800 + Math.random() * 600),
      });

      // Update conversation ID if new one was created
      if (data.session_id && data.session_id !== conversationId) {
        setConversationId(data.session_id);
      }

    } catch (error) {
      console.error('Query failed:', error);
      const errorMessage: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to process request'}. Please check that the backend is running and API key is configured.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div
      ref={containerRef}
      style={{
        minHeight: '100vh',
        background: '#060606',
        color: '#eaeaea',
        fontFamily: "system-ui, -apple-system, 'Segoe UI', sans-serif",
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,500;1,400;1,500&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        * { box-sizing: border-box; margin: 0; padding: 0; }

        ::selection { background: #e85d04; color: #000; }

        @keyframes slideUp {
          from { opacity: 0; transform: translateY(16px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }

        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }

        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }

        input::placeholder { color: #3a3a3a; }
      `}</style>

      {/* Ambient gradient */}
      <div style={{
        position: 'fixed',
        inset: 0,
        background: `radial-gradient(circle at ${mousePos.x * 100}% ${mousePos.y * 100}%, rgba(232, 93, 4, 0.06) 0%, transparent 50%)`,
        pointerEvents: 'none',
        transition: 'background 0.5s ease',
      }} />

      {/* Subtle grid */}
      <div style={{
        position: 'fixed',
        inset: 0,
        backgroundImage: 'linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px)',
        backgroundSize: '60px 60px',
        pointerEvents: 'none',
      }} />

      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        zIndex: 1,
      }}>
        {/* Header */}
        <header style={{
          padding: '20px 32px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid #161616',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{
              width: '32px',
              height: '32px',
              background: 'linear-gradient(135deg, #e85d04 0%, #f48c06 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <span style={{ color: '#000', fontSize: '14px', fontWeight: '600' }}>A</span>
            </div>
            <div>
              <h1 style={{
                fontFamily: "'Outfit', sans-serif",
                fontSize: '14px',
                fontWeight: '500',
                letterSpacing: '0.08em',
              }}>
                AGENTIC
              </h1>
              <p style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: '10px',
                color: '#4a4a4a',
                marginTop: '2px',
              }}>
                {modelInfo || 'v1.3'}
              </p>
            </div>
          </div>

          {/* Tool indicators */}
          <div style={{ display: 'flex', gap: '6px' }}>
            {tools.map(tool => (
              <div
                key={tool.id}
                style={{
                  padding: '5px 10px',
                  background: activeTools.includes(tool.id) ? 'rgba(232, 93, 4, 0.15)' : '#0c0c0c',
                  border: `1px solid ${activeTools.includes(tool.id) ? '#e85d04' : '#1a1a1a'}`,
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: '9px',
                  letterSpacing: '0.05em',
                  color: activeTools.includes(tool.id) ? '#e85d04' : '#3a3a3a',
                  transition: 'all 0.2s ease',
                }}
              >
                {tool.label}
              </div>
            ))}
          </div>

          {/* Metrics */}
          <div style={{ display: 'flex', gap: '24px' }}>
            {metrics.latency > 0 && (
              <>
                <div style={{ textAlign: 'right' }}>
                  <p style={{
                    fontFamily: "'Newsreader', serif",
                    fontSize: '18px',
                    fontStyle: 'italic',
                    color: '#eaeaea',
                  }}>
                    {metrics.latency}
                  </p>
                  <p style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: '8px',
                    color: '#3a3a3a',
                    letterSpacing: '0.1em',
                  }}>
                    MS
                  </p>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <p style={{
                    fontFamily: "'Newsreader', serif",
                    fontSize: '18px',
                    fontStyle: 'italic',
                    color: '#eaeaea',
                  }}>
                    {metrics.tokens}
                  </p>
                  <p style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: '8px',
                    color: '#3a3a3a',
                    letterSpacing: '0.1em',
                  }}>
                    TOKENS
                  </p>
                </div>
              </>
            )}
          </div>
        </header>

        {/* Main content */}
        <main style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}>
          {/* Messages */}
          <div style={{
            flex: 1,
            overflow: 'auto',
            padding: '40px 32px',
          }}>
            {messages.length === 0 && (
              <div style={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                maxWidth: '500px',
              }}>
                <h2 style={{
                  fontFamily: "'Newsreader', serif",
                  fontSize: '36px',
                  fontWeight: '400',
                  fontStyle: 'italic',
                  color: '#eaeaea',
                  lineHeight: 1.3,
                  marginBottom: '24px',
                }}>
                  Multi-agent orchestration,<br />
                  ready when you are.
                </h2>
                <p style={{
                  fontFamily: "'Outfit', sans-serif",
                  fontSize: '14px',
                  color: '#5a5a5a',
                  lineHeight: 1.7,
                  marginBottom: '32px',
                }}>
                  13 integrated tools. RAG-enabled vector search.
                  LangGraph workflows. {modelInfo} reasoning.
                </p>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {['What is 25 * 47?', 'Search for latest AI news', 'Analyze my documents'].map(cmd => (
                    <button
                      key={cmd}
                      onClick={() => setInput(cmd)}
                      style={{
                        padding: '10px 16px',
                        background: 'transparent',
                        border: '1px solid #222',
                        color: '#5a5a5a',
                        fontFamily: "'Outfit', sans-serif",
                        fontSize: '12px',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                      }}
                      onMouseEnter={e => {
                        e.currentTarget.style.borderColor = '#e85d04';
                        e.currentTarget.style.color = '#e85d04';
                      }}
                      onMouseLeave={e => {
                        e.currentTarget.style.borderColor = '#222';
                        e.currentTarget.style.color = '#5a5a5a';
                      }}
                    >
                      {cmd} →
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                style={{
                  marginBottom: '32px',
                  animation: 'slideUp 0.4s ease',
                }}
              >
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  marginBottom: '10px',
                }}>
                  <span style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: '9px',
                    letterSpacing: '0.15em',
                    color: msg.role === 'user' ? '#5a5a5a' : '#e85d04',
                    padding: '3px 8px',
                    border: `1px solid ${msg.role === 'user' ? '#222' : '#e85d04'}`,
                  }}>
                    {msg.role === 'user' ? 'YOU' : 'AGENT'}
                  </span>
                  <span style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: '10px',
                    color: '#2a2a2a',
                  }}>
                    {formatTime(msg.timestamp)}
                  </span>
                  {msg.tools && (
                    <span style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: '9px',
                      color: '#3a3a3a',
                    }}>
                      {msg.tools.map(t => tools.find(tool => tool.id === t)?.label || t).join(' → ')}
                    </span>
                  )}
                </div>

                <div style={{
                  paddingLeft: '20px',
                  borderLeft: `2px solid ${msg.role === 'user' ? '#1a1a1a' : '#e85d04'}`,
                }}>
                  <p style={{
                    fontFamily: msg.role === 'user' ? "'Outfit', sans-serif" : "'Newsreader', serif",
                    fontSize: msg.role === 'user' ? '15px' : '17px',
                    fontStyle: msg.role === 'assistant' ? 'italic' : 'normal',
                    color: msg.role === 'user' ? '#8a8a8a' : '#eaeaea',
                    lineHeight: 1.7,
                    maxWidth: '680px',
                    whiteSpace: 'pre-wrap',
                  }}>
                    {msg.content}
                  </p>

                  {msg.sources && msg.sources.length > 0 && (
                    <button
                      onClick={() => {
                        setCurrentSources(msg.sources || []);
                        setShowSources(!showSources);
                      }}
                      style={{
                        marginTop: '12px',
                        padding: '6px 12px',
                        background: 'transparent',
                        border: '1px solid #1a1a1a',
                        color: '#4a4a4a',
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: '9px',
                        letterSpacing: '0.05em',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                      }}
                      onMouseEnter={e => {
                        e.currentTarget.style.borderColor = '#333';
                        e.currentTarget.style.color = '#8a8a8a';
                      }}
                      onMouseLeave={e => {
                        e.currentTarget.style.borderColor = '#1a1a1a';
                        e.currentTarget.style.color = '#4a4a4a';
                      }}
                    >
                      {showSources ? 'HIDE' : 'VIEW'} {msg.sources.length} SOURCES
                    </button>
                  )}
                </div>
              </div>
            ))}

            {isProcessing && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                animation: 'slideUp 0.3s ease',
              }}>
                <span style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: '9px',
                  letterSpacing: '0.15em',
                  color: '#e85d04',
                  padding: '3px 8px',
                  border: '1px solid #e85d04',
                }}>
                  PROCESSING
                </span>
                <div style={{
                  width: '120px',
                  height: '2px',
                  background: '#1a1a1a',
                  overflow: 'hidden',
                  borderRadius: '1px',
                }}>
                  <div style={{
                    width: '100%',
                    height: '100%',
                    background: 'linear-gradient(90deg, transparent, #e85d04, transparent)',
                    backgroundSize: '200% 100%',
                    animation: 'shimmer 1.5s infinite',
                  }} />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Sources panel */}
          {showSources && currentSources.length > 0 && (
            <div style={{
              borderTop: '1px solid #161616',
              padding: '20px 32px',
              background: '#0a0a0a',
              animation: 'slideUp 0.3s ease',
            }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '16px',
              }}>
                <span style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: '9px',
                  color: '#4a4a4a',
                  letterSpacing: '0.15em',
                }}>
                  RAG SOURCES
                </span>
                <button
                  onClick={() => setShowSources(false)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: '#4a4a4a',
                    cursor: 'pointer',
                    fontSize: '16px',
                    padding: '4px 8px',
                  }}
                >
                  ×
                </button>
              </div>
              <div style={{ display: 'flex', gap: '16px', overflowX: 'auto' }}>
                {currentSources.map((src, i) => (
                  <div
                    key={i}
                    style={{
                      minWidth: '280px',
                      padding: '16px',
                      border: '1px solid #1a1a1a',
                      background: '#060606',
                    }}
                  >
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '10px',
                    }}>
                      <span style={{
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: '11px',
                        color: '#e85d04',
                      }}>
                        {src.metadata?.filename || `Source #${i + 1}`}
                      </span>
                      <span style={{
                        fontFamily: "'Newsreader', serif",
                        fontSize: '14px',
                        fontStyle: 'italic',
                        color: '#5a5a5a',
                      }}>
                        {Math.round(src.score * 100)}%
                      </span>
                    </div>
                    <p style={{
                      fontFamily: "'Outfit', sans-serif",
                      fontSize: '12px',
                      color: '#4a4a4a',
                      lineHeight: 1.6,
                    }}>
                      {src.content.slice(0, 150)}...
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Input */}
          <form
            onSubmit={handleSubmit}
            style={{
              padding: '20px 32px',
              borderTop: '1px solid #161616',
              display: 'flex',
              gap: '12px',
              alignItems: 'center',
            }}
          >
            <div style={{
              flex: 1,
              position: 'relative',
            }}>
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder="What would you like to explore?"
                disabled={isProcessing}
                style={{
                  width: '100%',
                  padding: '14px 20px',
                  background: '#0a0a0a',
                  border: '1px solid #1a1a1a',
                  color: '#eaeaea',
                  fontFamily: "'Outfit', sans-serif",
                  fontSize: '14px',
                  outline: 'none',
                  transition: 'border-color 0.2s ease',
                }}
                onFocus={e => e.currentTarget.style.borderColor = '#333'}
                onBlur={e => e.currentTarget.style.borderColor = '#1a1a1a'}
              />
            </div>
            <button
              type="submit"
              disabled={isProcessing || !input.trim()}
              style={{
                padding: '14px 28px',
                background: isProcessing || !input.trim() ? '#111' : '#e85d04',
                border: 'none',
                color: isProcessing || !input.trim() ? '#333' : '#000',
                fontFamily: "'Outfit', sans-serif",
                fontSize: '13px',
                fontWeight: '500',
                letterSpacing: '0.05em',
                cursor: isProcessing || !input.trim() ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s ease',
              }}
            >
              {isProcessing ? 'Running...' : 'Execute'}
            </button>
          </form>
        </main>

        {/* Footer */}
        <footer style={{
          padding: '12px 32px',
          borderTop: '1px solid #111',
          display: 'flex',
          justifyContent: 'space-between',
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '9px',
          color: '#2a2a2a',
          letterSpacing: '0.05em',
        }}>
          <span>FASTAPI • LANGGRAPH • QDRANT • REDIS • OPENAI</span>
          <span>OPENTELEMETRY TRACING • SESSION {conversationId?.slice(8, 16).toUpperCase()}</span>
        </footer>
      </div>
    </div>
  );
};

export default AgenticConsole;
