import { useEffect, useRef, useState } from 'react';

type Chunk = {
  id: number;
  text: string;
};

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isRecording) return;

    setChunks([]);

    async function startRecording() {
      // Open or reuse WebSocket
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        wsRef.current = new WebSocket("ws://localhost:8000/ws");
        wsRef.current.onmessage = (e) => {
          try {
            const { chunk_id, text } = JSON.parse(e.data);
            setChunks((prev) => [...prev, { id: chunk_id, text }]);
          } catch (err) {
            console.error("bad ws message", err);
          }
        };
      }
      const ws = wsRef.current!;

      // Audio context & worklet
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = ctx;

      await ctx.audioWorklet.addModule("/pcm-processor.js");
      const source = ctx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(ctx, "pcm-processor");
      workletNodeRef.current = workletNode;

      workletNode.port.onmessage = (event) => {
        const float32 = event.data as Float32Array;
        // Convert Float32 [-1,1] to Int16 PCM
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        ws.send(int16.buffer);
      };

      source.connect(workletNode).connect(ctx.destination);
    }

    startRecording();

    return () => {
      if (audioCtxRef.current) {
        audioCtxRef.current.close();
        audioCtxRef.current = null;
      }
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close(1000, "done");
      }
      workletNodeRef.current = null;
    };
  }, [isRecording]);

  // Auto‑scroll transcript to the bottom whenever a new chunk arrives
  useEffect(() => {
    const el = transcriptRef.current;
    if (!el) return;
    el.scrollTo({
      top: el.scrollHeight,
      behavior: 'smooth',
    });
  }, [chunks]);

  return (
    <div id="root">
      <h1>Voice Transcriber Demo</h1>

      <div className="controls">
        {isRecording ? (
          <button onClick={() => setIsRecording(false)}>Stop</button>
        ) : (
          <button onClick={() => setIsRecording(true)}>Record</button>
        )}
      </div>

      <div className="transcript" ref={transcriptRef}>
        {chunks.map((c) => (
          <span key={c.id}>{c.text} </span>
        ))}
      </div>
    </div>
  );
}
