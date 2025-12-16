import { useState, useRef, useEffect } from 'react'
import { Send, Cpu, Zap, Activity } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
}

interface StreamEvent {
  type: 'thought' | 'tool_start' | 'tool_result' | 'answer' | 'step' | 'error' | 'usage'
  content?: string
  name?: string
  args?: string
}

function App() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [events, setEvents] = useState<StreamEvent[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, events])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMsg: Message = { role: 'user', content: input, timestamp: new Date().toISOString() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)
    setEvents([]) // Clear previous thought stream

    try {
      const response = await fetch('/api/v1/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-key' // Replace with env
        },
        body: JSON.stringify({ question: userMsg.content, stream: true })
      })

      if (!response.ok) throw new Error(response.statusText)

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value)
          const lines = chunk.split('\n\n')

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6)
              if (data === '[DONE]') break

              try {
                const event = JSON.parse(data)
                setEvents(prev => [...prev, event])

                if (event.type === 'answer') {
                  // Only append final answer to chat history when complete? 
                  // or stream it? managing separate state for now.
                }
              } catch (e) {
                console.error('Parse error', e)
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [...prev, { role: 'system', content: `Error: ${error}`, timestamp: new Date().toISOString() }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex font-sans overflow-hidden">
      {/* Sidebar (Optional) */}
      <div className="w-64 bg-slate-950 border-r border-slate-800 p-4 hidden md:flex flex-col gap-4">
        <div className="flex items-center gap-2 text-xl font-bold text-cyan-400">
          <Cpu className="w-6 h-6" />
          <span>Antigravity</span>
        </div>
        <div className="flex-1 overflow-y-auto">
          {/* History items could go here */}
          <div className="text-sm text-slate-500 italic">No history</div>
        </div>
        <div className="p-2 bg-slate-900 rounded border border-slate-800 text-xs">
          <div className="flex items-center gap-2 text-green-400 mb-1">
            <Activity className="w-3 h-3" />
            <span>System Online</span>
          </div>
          <div className="text-slate-500">v1.0.0</div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative">
        {/* Header */}
        <div className="h-14 bg-slate-900/50 backdrop-blur border-b border-slate-800 flex items-center px-6 justify-between z-10">
          <h1 className="font-semibold text-slate-200">Orchestrator Console</h1>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Zap className="w-3 h-3" />
            <span>Streaming Enabled</span>
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
          {messages.map((msg, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-3xl p-4 rounded-2xl ${msg.role === 'user'
                  ? 'bg-cyan-600/20 border border-cyan-500/30 text-cyan-50'
                  : 'bg-slate-800/50 border border-slate-700/50 text-slate-200'
                }`}>
                <div className="whitespace-pre-wrap">{msg.content}</div>
              </div>
            </motion.div>
          ))}

          {/* Live Thought Stream */}
          <AnimatePresence>
            {events.length > 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex flex-col gap-2 max-w-3xl mr-auto w-full"
              >
                {events.map((event, idx) => {
                  if (event.type === 'step') return null;
                  if (event.type === 'thought') return (
                    <div key={idx} className="flex items-center gap-2 text-xs text-slate-500 font-mono pl-4">
                      <span className="w-1 h-1 bg-slate-500 rounded-full" />
                      Thinking... {event.content}
                    </div>
                  );
                  if (event.type === 'tool_start') return (
                    <div key={idx} className="bg-slate-950/50 border border-slate-800 rounded p-2 text-xs font-mono text-amber-500/80 ml-4 max-w-fit">
                      &gt; {event.name}({event.args?.slice(0, 50)}...)
                    </div>
                  );
                  if (event.type === 'answer') return (
                    <div key={idx} className="bg-slate-800/50 border border-slate-700/50 p-4 rounded-2xl text-slate-200 mt-2 whitespace-pre-wrap animate-pulse-once">
                      {event.content}
                    </div>
                  );
                  return null;
                })}
              </motion.div>
            )}
          </AnimatePresence>

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 bg-slate-900 border-t border-slate-800">
          <form onSubmit={handleSubmit} className="relative max-w-4xl mx-auto">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Assign a task to the swarm..."
              className="w-full bg-slate-950 border border-slate-800 rounded-xl py-3 px-4 pr-12 text-slate-200 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all placeholder:text-slate-600"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="absolute right-2 top-2 p-1.5 bg-cyan-600 text-white rounded-lg hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default App
