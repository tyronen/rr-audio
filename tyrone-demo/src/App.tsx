import { useEffect, useRef, useState } from 'react';

type Chunk = {
  id: number;
  text: string;
};

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunkIdRef = useRef(0);
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const sliceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // Start or stop recording whenever `isRecording` changes
    if (!isRecording) return;

      setChunks([]);
    async function startRecording() {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Decide which container the current browser supports
      let mimeType = "audio/ogg;codecs=opus";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = "audio/webm;codecs=opus";
      }

      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = async (evt) => {
        if (!evt.data || evt.data.size === 0) return;

        const chunkId = chunkIdRef.current++;
        const form = new FormData();
        const ext = mimeType.startsWith("audio/ogg") ? "ogg" : "webm";
        form.append("audio", evt.data, `chunk-${chunkId}.${ext}`);
        form.append('chunk_id', String(chunkId));

        try {
          const response = await fetch('http://localhost:8000/transcribe', {
            method: 'POST',
            body: form,
          });
          const { text } = await response.json();
          setChunks((prev) => [...prev, { id: chunkId, text }]);
        } catch (err) {
          console.error('transcription failed', err);
        }
      };

      recorder.start();
      const scheduleNext = () => {
          sliceTimer.current = setTimeout(() => {
              if (recorder.state === 'recording') {
                  recorder.stop();
              }
          }, 2_000);
      };
      scheduleNext();

      recorder.onstop = () => {
          if (sliceTimer.current) {
              clearTimeout(sliceTimer.current);
          }
          if (isRecording) {
              startRecording().catch(console.error);
          } else {
              recorder.stream.getTracks().forEach((t) => t.stop());
          }
      }
    }

    startRecording();

    return () => {
      // Gracefully stop the MediaRecorder and flush remaining data
      const rec = mediaRecorderRef.current;
      if (!rec) return;

      if (rec.state !== 'inactive') {
        // Ask for the last buffered data ( < 30 s )
        rec.requestData();

        // After recorder finishes emitting the final `dataavailable`,
        // release the underlying tracks.
        rec.onstop = () => {
          rec.stream.getTracks().forEach((t) => t.stop());
        };

        rec.stop();
      }
      if (sliceTimer.current) {
        clearTimeout(sliceTimer.current);
      }
      mediaRecorderRef.current = null;
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
