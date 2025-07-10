import { useEffect, useRef, useState } from 'react'
import './App.css'

type Segment = {
  id?: string
  chunk_id?: number
  start_sec: number
  end_sec: number
  text: string
}

export default function App() {
  const [isRecording, setIsRecording] = useState(false)
  const [segments, setSegments] = useState<Segment[]>([])
  const [recordedPCM, setRecordedPCM] = useState<Int16Array[]>([])
  const [editableTranscript, setEditableTranscript] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)

  const audioCtxRef = useRef<AudioContext | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const timerRef = useRef<number | null>(null)
  const transcriptRef = useRef<HTMLTextAreaElement | null>(null)

  // Start/stop timer
  useEffect(() => {
    if (isRecording) {
      setElapsed(0)
      timerRef.current = window.setInterval(() => setElapsed(e => e + 1), 1000)
    } else {
      if (timerRef.current) clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [isRecording])

  // Live transcription & PCM buffering
  useEffect(() => {
    if (!isRecording) return

    setSegments([])
    setRecordedPCM([])

    async function startRecording() {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        wsRef.current = new WebSocket('ws://localhost:8000/ws')
        wsRef.current.onmessage = e => {
          try {
            const { chunk_id, tokens, word_timestamps, start_sec, duration } = JSON.parse(e.data)
            const end = start_sec + duration
            const wordSegs = tokens.map((tok: string, i: number) => {
              const [s, t] = word_timestamps[i]
              return { id: `${chunk_id}-${i}`, start_sec: start_sec + s, end_sec: start_sec + t, text: tok }
            })
            setSegments(prev => {
              const filtered = prev.filter(s => s.end_sec <= start_sec || s.start_sec >= end)
              return [...filtered, ...wordSegs].sort((a, b) => a.start_sec - b.start_sec)
            })
          } catch(err) {
              console.error("bad ws message", err);
          }
        }
        wsRef.current.onclose = () => setIsProcessing(false)
      }
      const ws = wsRef.current!

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const ctx = new AudioContext({ sampleRate: 16000 })
      audioCtxRef.current = ctx
      await ctx.audioWorklet.addModule('/pcm-processor.js')
      const source = ctx.createMediaStreamSource(stream)
      const worklet = new AudioWorkletNode(ctx, 'pcm-processor')
      workletNodeRef.current = worklet

      worklet.port.onmessage = (e) => {
        const float32 = e.data as Float32Array
        const int16 = new Int16Array(float32.length)
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]))
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
        }
        ws.send(int16.buffer)
        setRecordedPCM(prev => [...prev, int16])
      }

      source.connect(worklet).connect(ctx.destination)
    }

    startRecording()

    return () => {
      audioCtxRef.current?.close()
      audioCtxRef.current = null
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(new ArrayBuffer(0))
        setIsProcessing(true)
      }
      workletNodeRef.current = null
    }
  }, [isRecording])

  // Build editable transcript
  useEffect(() => {
    setEditableTranscript(segments.map(s => s.text).join(""));
  }, [segments])

  // Auto-scroll
  useEffect(() => {
    const el = transcriptRef.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [segments])

  // Train handler
  const handleTrain = async () => {
    const total = recordedPCM.reduce((sum, a) => sum + a.length, 0)
    const allPCM = new Int16Array(total)
    let off = 0
    recordedPCM.forEach(arr => { allPCM.set(arr, off); off += arr.length })
    const blob = new Blob([allPCM.buffer], { type: 'audio/pcm' })
    const form = new FormData()
    form.append('audio', blob, 'audio.pcm')
    form.append('transcript', editableTranscript)

    const res = await fetch('http://localhost:8000/train', { method: 'POST', body: form })
    alert(res.ok ? 'Trained successfully!' : 'Train failed.')
  }

  return (
      <div className="app">
        <h1>Voice Transcriber Demo</h1>
        <div className="controls">
          <button
            onClick={() => setIsRecording(r => !r)}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              backgroundColor: isRecording ? '#c00' : '#fff',
              color: isRecording ? '#fff' : '#c00',
              border: '2px solid #c00',
              padding: '8px 12px',
              borderRadius: 4,
              marginRight: 16,
              cursor: 'pointer',
            }}
          >
            <span
              style={{
                width: 10,
                height: 10,
                backgroundColor: '#c00',
                borderRadius: isRecording ? 0 : '50%',
                display: 'inline-block',
                marginRight: 8,
              }}
            />
            {isRecording ? 'Stop' : 'Record'}
          </button>
          {/* Train green button */}
          <button
              disabled={isRecording || segments.length === 0}
              onClick={handleTrain}
              style={{
                backgroundColor: '#4caf50',
                color: '#fff',
                padding: '8px 16px',
                border: 'none',
                borderRadius: 4,
                cursor: segments.length > 0 ? 'pointer' : 'not-allowed',
              }}
          >
            Train
          </button>
          {/* Stopwatch */}
          <div style={{ marginLeft: 24, fontSize: "medium" }}>⏱ {elapsed}s</div>
          {/* Processing indicator */}
          {!isRecording && isProcessing && <div style={{ marginLeft: 24, fontStyle: 'italic' }}>Receiving more text…</div>}
        </div>
        <textarea
            ref={transcriptRef}
            value={editableTranscript}
            readOnly={isRecording}
            onChange={(e) => setEditableTranscript(e.target.value)}
            rows={24}
            style={{
              width: '100%',
              fontSize: '1.2em',
              marginTop: 16,
              backgroundColor: '#111',
              color: '#eee',
              padding: 12,
              borderRadius: 4,
              fontFamily: 'Courier, monospace',
            }}
        />
      </div>
  )
}