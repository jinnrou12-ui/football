import React, { useState, useRef, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [trackerColor, setTrackerColor] = useState('#FF0000');
  const [isDragOver, setIsDragOver] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null); // 'queued', 'processing', 'done', 'error'
  const [progress, setProgress] = useState(0);
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null);
  const [players, setPlayers] = useState([]);
  const [updatingNames, setUpdatingNames] = useState(false);
  const [toasts, setToasts] = useState([]);

  const fileInputRef = useRef(null);
  const pollIntervalRef = useRef(null);

  // Add toast helper
  const showToast = (message, type = 'success') => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  };

  // Color presets
  const presets = [
    { name: 'Red', hex: '#FF0000' },
    { name: 'Yellow', hex: '#FFEB3B' },
    { name: 'Cyan', hex: '#00E5FF' },
    { name: 'Green', hex: '#00E676' },
    { name: 'Magenta', hex: '#FF00FF' },
  ];

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile) => {
    const suffix = selectedFile.name.substring(selectedFile.name.lastIndexOf('.')).toLowerCase();
    const allowed = ['.mp4', '.mov', '.avi', '.mkv'];
    if (!allowed.includes(suffix)) {
      showToast('Unsupported video format. Please upload .mp4, .mov, .avi, or .mkv.', 'error');
      return;
    }
    setFile(selectedFile);
    setFileUrl(URL.createObjectURL(selectedFile));
    setProcessedVideoUrl(null);
    setJobId(null);
    setJobStatus(null);
    setProgress(0);
    setPlayers([]);
    showToast('Video loaded successfully.');
  };

  // Programmatically load the sample match video from public folder to test the system
  const handleLoadSampleMatch = async (e) => {
    e.stopPropagation(); // prevent triggering upload-zone file dialog click
    try {
      showToast('Downloading sample match video...');
      const response = await fetch('/sample_football.mp4');
      if (!response.ok) throw new Error('Sample video not found in public folder');
      const blob = await response.blob();
      const sampleFile = new File([blob], 'sample_football.mp4', { type: 'video/mp4' });
      setFile(sampleFile);
      setFileUrl(URL.createObjectURL(sampleFile));
      setProcessedVideoUrl(null);
      setJobId(null);
      setJobStatus(null);
      setProgress(0);
      setPlayers([]);
      showToast('Sample match video loaded.');
    } catch (err) {
      console.error(err);
      showToast('Failed to load sample match: ' + err.message, 'error');
    }
  };

  // Start processing
  const handleProcessVideo = async () => {
    if (!file) return;

    try {
      setProcessing(true);
      setJobStatus('queued');
      setProgress(0);
      setProcessedVideoUrl(null);
      setPlayers([]);

      const formData = new FormData();
      formData.append('video', file);
      formData.append('tracker_color', trackerColor);

      const response = await fetch(`${API_BASE_URL}/upload-video`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload video');
      }

      const data = await response.json();
      setJobId(data.job_id);
      showToast('Processing started in background.');
    } catch (err) {
      console.error(err);
      setProcessing(false);
      setJobStatus('error');
      showToast('Error uploading video: ' + err.message, 'error');
    }
  };

  // Poll job status
  useEffect(() => {
    if (!jobId) return;

    pollIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/job-status/${jobId}`);
        if (!response.ok) {
          throw new Error('Failed to fetch job status');
        }
        const data = await response.json();
        setJobStatus(data.status);
        setProgress(data.progress || 0);

        if (data.status === 'done') {
          clearInterval(pollIntervalRef.current);
          setProcessing(false);
          // Set streaming URL
          setProcessedVideoUrl(`${API_BASE_URL}/stream-video/${data.filename}`);
          setPlayers(data.players || []);
          showToast('Video processed successfully!', 'success');
        } else if (data.status === 'error') {
          clearInterval(pollIntervalRef.current);
          setProcessing(false);
          showToast('Backend processing error: ' + (data.error || 'unknown'), 'error');
        }
      } catch (err) {
        console.error(err);
        clearInterval(pollIntervalRef.current);
        setProcessing(false);
        setJobStatus('error');
        showToast('Polling error: ' + err.message, 'error');
      }
    }, 1500);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [jobId]);

  const handlePlayerNameChange = (trackId, newName) => {
    setPlayers((prev) =>
      prev.map((p) => (p.track_id === trackId ? { ...p, name: newName } : p))
    );
  };

  const handleUpdatePlayerNames = async () => {
    if (!jobId) return;
    try {
      setUpdatingNames(true);
      const namesMap = {};
      players.forEach((p) => {
        namesMap[p.track_id] = p.name;
      });

      const response = await fetch(`${API_BASE_URL}/update-player-names/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ names: namesMap }),
      });

      if (!response.ok) {
        throw new Error('Failed to update player names');
      }

      const data = await response.json();
      setProcessedVideoUrl(`${API_BASE_URL}/stream-video/${data.filename}?t=${Date.now()}`);
      setPlayers(data.players || []);
      showToast('Player names updated successfully!');
    } catch (err) {
      console.error(err);
      showToast('Error updating names: ' + err.message, 'error');
    } finally {
      setUpdatingNames(false);
    }
  };

  return (
    <div className="app-shell">
      {/* Toast container */}
      <div className="toast-container">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            <span className="toast-icon">{toast.type === 'success' ? '⚡' : '⚠️'}</span>
            <div className="toast-msg">{toast.message}</div>
          </div>
        ))}
      </div>

      {/* Header */}
      <header className="header">
        <div className="header-icon">⚽</div>
        <div>
          <h1 className="header-title">Football Analysis Studio</h1>
          <div className="header-subtitle">AI-Powered Tactical Video Tracker & Vision Engine</div>
        </div>
        <div className="header-badge">YOLOv8 Engine</div>
      </header>

      {/* Main workspace */}
      <div className="main-grid">
        {/* Sidebar Controls */}
        <aside className="panel">
          <div className="panel-header">
            <div className="panel-icon">⚙️</div>
            <h2 className="panel-title">Studio Controls</h2>
          </div>
          <div className="panel-body">
            {/* Upload Zone */}
            <div
              className={`upload-zone ${isDragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current && fileInputRef.current.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                className="upload-input"
                accept="video/*"
                onChange={handleFileChange}
              />
              <span className="upload-icon">📤</span>
              <div className="upload-title">
                {file ? 'Replace football video' : 'Upload match video'}
              </div>
              <div className="upload-sub">Drag & drop or click to browse</div>
              {file && <div className="upload-file-name">{file.name}</div>}
            </div>

            {/* Load Demo Match Video Button */}
            <button
              onClick={handleLoadSampleMatch}
              style={{
                width: '100%',
                marginTop: '12px',
                padding: '12px',
                border: '1px solid var(--border-active)',
                borderRadius: 'var(--radius-sm)',
                background: 'rgba(56, 189, 248, 0.06)',
                color: 'var(--accent)',
                fontSize: '12px',
                fontWeight: '700',
                fontFamily: "'Outfit', sans-serif",
                cursor: 'pointer',
                transition: 'background 0.2s, transform 0.1s',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px'
              }}
              onMouseEnter={(e) => e.target.style.background = 'rgba(56, 189, 248, 0.12)'}
              onMouseLeave={(e) => e.target.style.background = 'rgba(56, 189, 248, 0.06)'}
            >
              🎬 Load Demo Match Video
            </button>

            {/* Custom Tracker Color Selection */}
            <div className="color-section">
              <span className="section-label">Ball Tracker Color</span>
              <div className="color-presets">
                {presets.map((preset) => (
                  <button
                    key={preset.name}
                    className={`color-preset ${trackerColor === preset.hex ? 'selected' : ''}`}
                    style={{ backgroundColor: preset.hex, color: preset.hex }}
                    onClick={() => setTrackerColor(preset.hex)}
                    title={preset.name}
                  />
                ))}
              </div>
              <div className="color-custom-row">
                <input
                  type="color"
                  className="color-picker-native"
                  value={trackerColor}
                  onChange={(e) => setTrackerColor(e.target.value)}
                />
                <span className="color-hex-label">{trackerColor.toUpperCase()}</span>
              </div>
            </div>

            {/* Status Display */}
            {jobStatus && (
              <div style={{ marginTop: '20px', borderTop: '1px solid var(--border)', paddingTop: '15px' }}>
                <span className="section-label">Pipeline Status</span>
                <div>
                  <span className={`status-badge ${jobStatus}`}>
                    <span className="status-dot"></span>
                    {jobStatus}
                  </span>
                </div>

                {processing && (
                  <>
                    <div className="progress-wrap">
                      <div className="progress-bar" style={{ width: `${progress}%` }}></div>
                    </div>
                    <div className="progress-label">Processing: {progress}%</div>
                  </>
                )}
              </div>
            )}

            {/* Action button */}
            <button
              className="process-btn"
              disabled={!file || processing}
              onClick={handleProcessVideo}
            >
              {processing ? (
                <>
                  <div className="spinner"></div>
                  Analyzing Frames...
                </>
              ) : (
                <>
                  <span>⚡</span> Run Visual Engine
                </>
              )}
            </button>

            {/* Feature lists */}
            <div className="feature-list">
              <span className="section-label" style={{ marginTop: '10px' }}>Active CV Layers</span>
              <div className="feature-item">
                <span className="feature-item-icon">⭕</span>
                <span>Custom Ball Tracker Overlay</span>
              </div>
              <div className="feature-item">
                <span className="feature-item-icon">👤</span>
                <span>Possession Proximity Detection</span>
              </div>
              <div className="feature-item">
                <span className="feature-item-icon">🌫️</span>
                <span>Non-Possession Background Blur</span>
              </div>
            </div>
          </div>
        </aside>

        {/* Video Player Display Panels */}
        <main className="video-panel-area">
          <div className="videos-grid">
            {/* Raw video container */}
            <div className="video-card">
              <div className="video-card-header">
                <span className="video-card-label">
                  <span className="video-card-dot live"></span>
                  Original Source
                </span>
              </div>
              <div className="video-card-body">
                {fileUrl ? (
                  <video src={fileUrl} className="video-el" controls playsInline />
                ) : (
                  <div className="video-placeholder">
                    <span className="video-placeholder-icon">📼</span>
                    <span className="video-placeholder-text">Please upload a source match video</span>
                  </div>
                )}
              </div>
            </div>

            {/* Processed video container */}
            <div className="video-card">
              <div className="video-card-header">
                <span className="video-card-label">
                  <span className="video-card-dot processed"></span>
                  Computer Vision Output
                </span>
                {processedVideoUrl && (
                  <a
                    href={`${API_BASE_URL}/download-video/processed_${jobId}.mp4`}
                    download
                    className="download-btn"
                  >
                    📥 Download
                  </a>
                )}
              </div>
              <div className="video-card-body">
                {processing ? (
                  <div className="processing-overlay">
                    <div className="processing-spinner-large"></div>
                    <span className="processing-text">
                      Running YOLOv8s Tracker ({progress}%)
                    </span>
                  </div>
                ) : processedVideoUrl ? (
                  <video src={processedVideoUrl} className="video-el" controls playsInline autoPlay loop />
                ) : (
                  <div className="video-placeholder">
                    <span className="video-placeholder-icon">🤖</span>
                    <span className="video-placeholder-text">Run Visual Engine to generate output</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Tactical dashboard values */}
          <div className="stats-row">
            <div className="stat-card">
              <div className="stat-value">YOLOv8s</div>
              <div className="stat-label">Model Architecture</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">&lt; 80px</div>
              <div className="stat-label">Proximity Threshold</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">CV2 Blur</div>
              <div className="stat-label">Background Mask</div>
            </div>
          </div>

          {/* Player Directory Card */}
          {players.length > 0 && (
            <div className="panel" style={{ marginTop: '24px' }}>
              <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div className="panel-icon">👥</div>
                  <h2 className="panel-title">Player Directory & Tag Manager</h2>
                </div>
                <button
                  className="update-names-btn"
                  disabled={updatingNames}
                  onClick={handleUpdatePlayerNames}
                  style={{
                    padding: '8px 18px',
                    border: 'none',
                    borderRadius: 'var(--radius-sm)',
                    background: 'linear-gradient(135deg, var(--accent-dark), var(--accent))',
                    color: '#fff',
                    fontWeight: '700',
                    fontSize: '12px',
                    fontFamily: "'Outfit', sans-serif",
                    cursor: 'pointer',
                    transition: 'transform 0.1s, box-shadow 0.2s',
                    boxShadow: '0 2px 10px var(--accent-glow)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}
                >
                  {updatingNames ? (
                    <>
                      <div className="spinner" style={{ width: '12px', height: '12px', borderWidth: '1px' }}></div>
                      Applying...
                    </>
                  ) : (
                    <>
                      <span>🏷️</span> Apply Names
                    </>
                  )}
                </button>
              </div>
              <div className="panel-body">
                <p style={{ color: 'var(--text-secondary)', fontSize: '13px', marginBottom: '20px' }}>
                  We detected <strong>{players.length} players</strong> in the video. Assign custom names below and click <strong>Apply Names</strong> to instantly update the video highlights.
                </p>
                
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
                  gap: '16px'
                }}>
                  {players.map((player) => (
                    <div key={player.track_id} style={{
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-md)',
                      padding: '12px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      transition: 'border-color 0.2s, background 0.2s'
                    }}
                    className="player-card-item"
                    >
                      {/* Thumbnail Avatar */}
                      <div style={{
                        width: '60px',
                        height: '80px',
                        borderRadius: 'var(--radius-sm)',
                        overflow: 'hidden',
                        background: '#000',
                        border: '1px solid var(--border)',
                        flexShrink: 0,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <img
                          src={`${API_BASE_URL}/thumbnail/${player.thumbnail}`}
                          alt={`P${player.track_id}`}
                          style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover'
                          }}
                          onError={(e) => {
                            e.target.onerror = null;
                            e.target.src = 'https://via.placeholder.com/60x80/000000/FFFFFF?text=Player';
                          }}
                        />
                      </div>
                      
                      {/* Input form */}
                      <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '11px', fontWeight: '800', color: 'var(--accent)', textTransform: 'uppercase' }}>
                            ID #{player.track_id}
                          </span>
                          <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                            Detected Track
                          </span>
                        </div>
                        <input
                          type="text"
                          value={player.name.startsWith("Player ") ? (player.name === `Player ${player.track_id}` ? "" : player.name) : player.name}
                          placeholder={player.name}
                          onChange={(e) => handlePlayerNameChange(player.track_id, e.target.value || `Player ${player.track_id}`)}
                          style={{
                            width: '100%',
                            padding: '8px 10px',
                            border: '1px solid var(--border)',
                            borderRadius: 'var(--radius-sm)',
                            background: 'var(--bg-panel)',
                            color: 'var(--text-primary)',
                            fontSize: '13px',
                            fontFamily: "'Inter', sans-serif",
                            transition: 'border-color 0.25s, box-shadow 0.25s'
                          }}
                          className="player-names-input"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
