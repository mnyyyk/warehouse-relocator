import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';

/**
 * Generic component to upload a single file and POST it to `endpoint`.
 * Displays upload progress and the server’s JSON/plain‑text response.
 */
const UploadSection = ({ title, endpoint }) => {
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [msg, setMsg] = useState('');

  const handleChange = (e) => {
    setFile(e.target.files[0]);
    setProgress(0);
    setMsg('');
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', endpoint);

    // Progress for nicer UX
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) {
        setProgress(Math.round((ev.loaded / ev.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          setMsg(JSON.stringify(JSON.parse(xhr.responseText), null, 2));
        } catch {
          setMsg(xhr.responseText || 'Success');
        }
      } else {
        setMsg(xhr.responseText || 'Upload failed');
      }
      setProgress(0);
    };

    xhr.onerror = () => {
      setMsg('Network error');
      setProgress(0);
    };

    xhr.send(formData);
  };

  return (
    <section>
      <h2>{title}</h2>
      <form onSubmit={handleSubmit}>
        <label htmlFor={`${endpoint}-file`}>Choose file:</label>
        <input
          id={`${endpoint}-file`}
          type="file"
          accept=".xlsx,.csv"
          onChange={handleChange}
        />
        <button type="submit" disabled={!file}>
          Upload
        </button>
      </form>

      {progress > 0 && (
        <div className="progress">
          <div className="progress-bar" style={{ width: `${progress}%` }} />
        </div>
      )}

      {msg && <pre className="result">{msg}</pre>}
    </section>
  );
};

function App() {
  return (
    <div className="container">
      <h1>Warehouse Relocator</h1>

      {/* Master / transaction files */}
      <UploadSection
        title="Upload master & transaction data"
        endpoint="/api/uploads"
      />

      {/* Inventory optimisation */}
      <UploadSection
        title="Run relocation on current inventory"
        endpoint="/api/optimize"
      />
    </div>
  );
}

const root = createRoot(document.getElementById('root'));
root.render(<App />);
