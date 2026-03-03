import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, ImageOverlay, useMap } from 'react-leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix Leaflet Icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const DEFAULT_ZOOM = 13.5;
const BACKEND_URL = 'https://satvision-app.hf.space';

// ────────────────────────────────────────────────
// Image Modal Component
// ────────────────────────────────────────────────
const ImageModal = ({ url, date, title, source, onClose }) => {
  if (!url) return null;
  const freshUrl = `${url}?t=${new Date().getTime()}`;

  return (
    <div
      style={{
        position: 'fixed', inset: 0,
        backgroundColor: 'rgba(0,0,0,0.85)', zIndex: 9999,
        display: 'flex', justifyContent: 'center', alignItems: 'center',
        backdropFilter: 'blur(5px)'
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#161b22', padding: '20px', borderRadius: '12px',
          width: '90%', maxWidth: '800px', maxHeight: '90%',
          display: 'flex', flexDirection: 'column', gap: '15px',
          border: '1px solid #30363d'
        }}
        onClick={e => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #30363d', paddingBottom: '10px' }}>
          <div>
            <h3 style={{ margin: 0, color: '#e6edf3' }}>{title}</h3>
            <div style={{ fontSize: '13px', color: '#8b949e', marginTop: '4px' }}>
              📅 Capture Date: <b>{date || '—'}</b>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{ border: 'none', background: '#21262d', color: '#c9d1d9', borderRadius: '50%', width: '32px', height: '32px', cursor: 'pointer', fontSize: '16px' }}
          >
            ✕
          </button>
        </div>

        <div style={{ flex: 1, overflow: 'hidden', display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#000', borderRadius: '8px' }}>
          <img
            src={freshUrl}
            alt="Satellite Source"
            style={{ maxWidth: '100%', maxHeight: '60vh', objectFit: 'contain' }}
            onError={() => console.error('Image failed to load:', freshUrl)}
          />
        </div>

        <div style={{ fontSize: '12px', color: '#8b949e', textAlign: 'center' }}>
          Source: {source || "Sentinel-2"}
        </div>
      </div>
    </div>
  );
};

// ────────────────────────────────────────────────
// Map Helpers
// ────────────────────────────────────────────────
const MapController = ({ setBounds }) => {
  const map = useMap();
  useEffect(() => {
    const onMove = () => setBounds(map.getBounds());
    map.on('moveend', onMove);
    setBounds(map.getBounds());
    return () => map.off('moveend', onMove);
  }, [map, setBounds]);
  return null;
};

const MapFlyTo = ({ targetLocation }) => {
  const map = useMap();
  useEffect(() => {
    if (targetLocation) map.flyTo(targetLocation, DEFAULT_ZOOM, { animate: true, duration: 1.5 });
  }, [targetLocation, map]);
  return null;
};

// ────────────────────────────────────────────────
// MAIN APP
// ────────────────────────────────────────────────
const App = () => {
  const [mapBounds, setMapBounds] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusLog, setStatusLog] = useState("");
  const [errorMsg, setErrorMsg] = useState(null);
  const [targetLocation, setTargetLocation] = useState(null);
  const [overlayBounds, setOverlayBounds] = useState(null);

  // Search
  const [searchQuery, setSearchQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const searchTimeoutRef = useRef(null);

  // Results
  const [layers, setLayers] = useState({ latest: null, previous: null, flood: null });
  const [meta, setMeta] = useState({
    latest_date: "", previous_date: "",
    latest_source: "", previous_source: "",
    latest_rgb: null, previous_rgb: null
  });
  const [reportData, setReportData] = useState(null);

  // UI toggles
  const [showCurrentWater, setShowCurrentWater] = useState(true);
  const [showHistoricWater, setShowHistoricWater] = useState(true);
  const [showFlood, setShowFlood] = useState(true);
  const [showActualImage, setShowActualImage] = useState(true);
  const [modalData, setModalData] = useState(null);

  // Responsiveness State
  const [isMobile, setIsMobile] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [layerControlCollapsed, setLayerControlCollapsed] = useState(false);

  // Dragging State for Control Panel
  const [panelOffset, setPanelOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragInfo = useRef({ startX: 0, startY: 0, isDragging: false });

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth <= 768;
      setIsMobile(mobile);
      setSidebarOpen(!mobile);
      if (mobile) setLayerControlCollapsed(true);
    };
    
    handleResize(); 
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // ── Drag Handlers ──────────────────────────────────────
  const handlePointerDown = (e) => {
    dragInfo.current.isDragging = true;
    dragInfo.current.startX = e.clientX - panelOffset.x;
    dragInfo.current.startY = e.clientY - panelOffset.y;
    setIsDragging(true);
    e.target.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e) => {
    if (!dragInfo.current.isDragging) return;
    setPanelOffset({
      x: e.clientX - dragInfo.current.startX,
      y: e.clientY - dragInfo.current.startY
    });
  };

  const handlePointerUp = (e) => {
    dragInfo.current.isDragging = false;
    setIsDragging(false);
    e.target.releasePointerCapture(e.pointerId);
  };
  // ── Secure Downloader ───────────────────────────────
  const handleSecureDownload = async (url, filename) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error("Download failed, opening in new tab...", error);
      window.open(url, '_blank');
    }
  };

  // ── Search logic ──────────────────────────────────────
  const fetchSuggestions = async (query) => {
    try {
      const res = await axios.get(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&countrycodes=pk&limit=5`
      );
      setSuggestions(res.data);
      setShowSuggestions(true);
    } catch (err) {
      console.error("Suggestion fetch failed", err);
    }
  };

  const handleInputChange = (e) => {
    const val = e.target.value;
    setSearchQuery(val);

    if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current);

    if (val.length > 2) {
      searchTimeoutRef.current = setTimeout(() => fetchSuggestions(val), 500);
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  };

  const handleSuggestionSelect = (suggestion) => {
    setSearchQuery(suggestion.display_name);
    setTargetLocation([parseFloat(suggestion.lat), parseFloat(suggestion.lon)]);
    setShowSuggestions(false);
    setSuggestions([]);
  };

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!searchQuery) return;
    try {
      const res = await axios.get(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&countrycodes=pk`
      );
      if (res.data?.[0]) {
        setTargetLocation([parseFloat(res.data[0].lat), parseFloat(res.data[0].lon)]);
        setShowSuggestions(false);
      } else {
        setErrorMsg("Location not found in Pakistan.");
      }
    } catch {
      setErrorMsg("Search failed.");
    }
  };

  // ── Flood Detection ───────────────────────────────
  const handleDetectFloods = async () => {
    if (!mapBounds) {
      setErrorMsg("Zoom/move the map first to select an area.");
      return;
    }

    setIsProcessing(true);
    setErrorMsg(null);
    setProgress(0);
    setStatusLog("Connecting to analysis server...");
    setLayers({ latest: null, previous: null, flood: null });
    setReportData(null);

    const currentBounds = mapBounds;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000);

    try {
      const bbox = {
        north: currentBounds.getNorth(),
        south: currentBounds.getSouth(),
        east: currentBounds.getEast(),
        west: currentBounds.getWest(),
      };

      const response = await fetch(`${BACKEND_URL}/api/detect_stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bbox }),
        signal: controller.signal,
      });

      if (!response.ok) throw new Error(`Server responded ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        let lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (let rawLine of lines) {
          const line = rawLine.trim();
          if (!line) continue;

          try {
            const data = JSON.parse(line);

            if (data.progress !== undefined) setProgress(data.progress);
            if (data.log) setStatusLog(data.log);
            if (data.error) throw new Error(data.error);

            if (data.report) {
              setReportData(data.report);
            }

            if (data.result) {
              setOverlayBounds(currentBounds);
              setLayers({
                latest: data.result.latest,
                previous: data.result.previous,
                flood: data.result.flood,
              });
              setMeta(data.meta || {});
              if (isMobile) setLayerControlCollapsed(false); 
            }
          } catch (parseErr) {
            console.error("Parse error:", parseErr);
          }
        }
      }
      setStatusLog("Analysis complete!");
    } catch (err) {
      setErrorMsg(err.name === 'AbortError' ? "Request timed out." : err.message);
    } finally {
      clearTimeout(timeoutId);
      setIsProcessing(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', backgroundColor: '#0d1117', position: 'relative' }}>
      
      {/* Navbar */}
      <div style={{ height: '60px', backgroundColor: '#161b22', borderBottom: '1px solid #30363d', display: 'flex', alignItems: 'center', padding: '0 20px', gap: '20px', flexShrink: 0 }}>
        {isMobile && (
          <div style={{ color: '#20b2aa', fontSize: '24px', cursor: 'pointer' }} onClick={() => setSidebarOpen(!sidebarOpen)}>≡</div>
        )}
        <h2 style={{ margin: 0, color: '#e6edf3', fontSize: isMobile ? '18px' : '20px', fontWeight: '600', whiteSpace: 'nowrap' }}>
          SATVision { !isMobile && <span style={{ color: '#8b949e', fontWeight: '400' }}>| Map Explorer</span> }
        </h2>
        
        {!isMobile && (
          <div style={{ display: 'flex', gap: '20px', marginLeft: '30px' }}>
            <span style={{ color: '#c9d1d9', cursor: 'pointer', fontSize: '14px' }}>Dashboard</span>
            <span style={{ color: '#8b949e', cursor: 'pointer', fontSize: '14px' }}>Reports</span>
            <span style={{ color: '#8b949e', cursor: 'pointer', fontSize: '14px' }}>Settings</span>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', position: 'relative' }}>
        
        {isMobile && sidebarOpen && (
          <div style={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)', zIndex: 2999 }} onClick={() => setSidebarOpen(false)} />
        )}
        
        
        {/* Sidebar */}
        <div style={{ 
          width: isMobile ? '280px' : '320px', 
          backgroundColor: '#161b22', borderRight: '1px solid #30363d', 
          padding: '20px', display: 'flex', flexDirection: 'column', gap: '30px', overflowY: 'auto',
          position: isMobile ? 'absolute' : 'relative', height: '100%',
          left: isMobile ? (sidebarOpen ? '0' : '-100%') : '0',
          zIndex: 3000, transition: 'left 0.3s ease',
          boxShadow: isMobile && sidebarOpen ? '4px 0 15px rgba(0,0,0,0.5)' : 'none'
        }}>
          <div>
            <h3 style={{ color: '#e6edf3', borderBottom: '2px solid #20b2aa', paddingBottom: '10px', marginBottom: '15px' }}>AI Assessment</h3>
            
            {reportData ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                {/* Metrics Box */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <div style={{ backgroundColor: '#21262d', padding: '12px', borderRadius: '8px', border: '1px solid #30363d', textAlign: 'center' }}>
                    <div style={{ color: '#8b949e', fontSize: '11px', marginBottom: '4px' }}>New Flooded Area</div>
                    <div style={{ color: '#ff4d4f', fontSize: '16px', fontWeight: 'bold' }}>
                      {reportData.metrics?.flood_sq_km?.toFixed(2) || '0.00'} km²
                    </div>
                  </div>
                  <div style={{ backgroundColor: '#21262d', padding: '12px', borderRadius: '8px', border: '1px solid #30363d', textAlign: 'center' }}>
                    <div style={{ color: '#8b949e', fontSize: '11px', marginBottom: '4px' }}>Current Total Water</div>
                    <div style={{ color: '#1890ff', fontSize: '16px', fontWeight: 'bold' }}>
                      {reportData.metrics?.current_sq_km?.toFixed(2) || '0.00'} km²
                    </div>
                  </div>
                </div>

                {/* LLM Text Report */}
                <div style={{ color: '#c9d1d9', fontSize: '13px', lineHeight: '1.6', whiteSpace: 'pre-wrap', maxHeight: '400px', overflowY: 'auto', padding: '15px', backgroundColor: '#21262d', borderRadius: '8px', border: '1px solid #30363d' }}>
                  {reportData.text}
                </div>

                {/* Download Button */}
                <a 
                  href={`${reportData.download_url}?t=${Date.now()}`}
                  download="SatVision_Flood_Report.pdf"
                  style={{ backgroundColor: '#20b2aa', color: '#0d1117', padding: '12px', borderRadius: '8px', textAlign: 'center', textDecoration: 'none', fontSize: '14px', fontWeight: '600', display: 'block', transition: 'opacity 0.2s' }}
                  onMouseEnter={e => e.currentTarget.style.opacity = 0.8}
                  onMouseLeave={e => e.currentTarget.style.opacity = 1}
                >
                  📥 Download Full Report
                </a>
              </div>
            ) : (
              <div style={{ backgroundColor: '#21262d', padding: '20px', borderRadius: '8px', border: '1px dashed #30363d', color: '#8b949e', fontSize: '13px', textAlign: 'center', lineHeight: '1.5' }}>
                🤖 Run an area analysis to generate an AI assessment report and metrics.
              </div>
            )}
          </div>
        </div>

        {/* Main Content */}
        <div style={{ flex: 1, padding: isMobile ? '10px' : '20px', display: 'flex', flexDirection: 'column', gap: '15px', backgroundColor: '#0d1117', position: 'relative' }}>
          
          <div style={{ display: 'flex', gap: '10px', flexDirection: isMobile ? 'column' : 'row' }}>
            <form onSubmit={handleSearch} style={{ flex: 1, position: 'relative' }}>
              <input 
                style={{ width: '100%', padding: '15px 20px', backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '8px', color: '#c9d1d9', fontSize: '16px', outline: 'none', boxSizing: 'border-box' }} 
                placeholder="Search location (e.g., Lahore)..." 
                value={searchQuery} onChange={handleInputChange}
                onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
                onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
              />
              {showSuggestions && suggestions.length > 0 && (
                <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '8px', marginTop: '5px', zIndex: 3000, overflow: 'hidden' }}>
                  {suggestions.map((s, i) => (
                    <div key={i} onMouseDown={() => handleSuggestionSelect(s)} style={{ padding: '12px 20px', color: '#c9d1d9', cursor: 'pointer', borderBottom: '1px solid #30363d', fontSize: '14px' }}>
                      {s.display_name}
                    </div>
                  ))}
                </div>
              )}
            </form>
            <button 
              onClick={handleDetectFloods} disabled={isProcessing}
              style={{ backgroundColor: '#20b2aa', color: '#0d1117', border: 'none', padding: '15px 30px', borderRadius: '8px', fontSize: '16px', fontWeight: '600', cursor: isProcessing ? 'not-allowed' : 'pointer', width: isMobile ? '100%' : 'auto' }}
            >
              {isProcessing ? 'Processing...' : 'Analyse Area'}
            </button>
          </div>

          {errorMsg && <div style={{ color: '#ff7b72', backgroundColor: 'rgba(255,123,114,0.1)', padding: '10px', borderRadius: '8px', border: '1px solid rgba(255,123,114,0.4)' }}>{errorMsg}</div>}

          {/* Map Area */}
          <div style={{ flex: 1, borderRadius: '12px', overflow: 'hidden', border: '1px solid #30363d', position: 'relative' }}>
            
            {isProcessing && (
              <div style={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(13,17,23,0.8)', zIndex: 2000, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                <h3 style={{ color: '#e6edf3', marginBottom: '15px' }}>Analyzing Satellite Data</h3>
                <div style={{ width: '80%', maxWidth: '300px', height: '8px', backgroundColor: '#21262d', borderRadius: '4px', overflow: 'hidden', marginBottom: '15px' }}>
                  <div style={{ width: `${progress}%`, height: '100%', backgroundColor: '#20b2aa', transition: 'width 0.3s ease' }}></div>
                </div>
                <div style={{ color: '#8b949e', fontSize: '14px' }}>{statusLog}</div>
              </div>
            )}

            {/* Collapsed Mobile Button */}
            {overlayBounds && !isProcessing && isMobile && layerControlCollapsed && (
              <button 
                onClick={() => setLayerControlCollapsed(false)}
                style={{ position: 'absolute', bottom: '20px', right: '10px', zIndex: 1000, backgroundColor: '#20b2aa', color: '#0d1117', border: 'none', borderRadius: '50px', padding: '12px 20px', fontWeight: 'bold', boxShadow: '0 4px 12px rgba(0,0,0,0.5)', cursor: 'pointer' }}
              >
                🎛️ Layers
              </button>
            )}

            {/* Draggable Layer Controls */}
            {overlayBounds && !isProcessing && (!isMobile || !layerControlCollapsed) && (
              <div style={{ 
                position: 'absolute', top: isMobile ? 'auto' : '20px', bottom: isMobile ? '20px' : 'auto', right: isMobile ? '10px' : '20px', left: isMobile ? '10px' : 'auto',
                zIndex: 1000, backgroundColor: 'rgba(22, 27, 34, 0.95)', border: '1px solid #30363d', padding: '15px', borderRadius: '8px', 
                display: 'flex', flexDirection: 'column', gap: '12px', minWidth: '240px', boxShadow: '0 8px 24px rgba(0,0,0,0.5)', backdropFilter: 'blur(5px)',
                transform: `translate(${panelOffset.x}px, ${panelOffset.y}px)`
              }}>
                <div 
                  onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={handlePointerUp} onPointerCancel={handlePointerUp}
                  style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #30363d', paddingBottom: '8px', cursor: isDragging ? 'grabbing' : 'grab', touchAction: 'none' }}
                >
                  <h4 style={{ margin: 0, color: '#e6edf3', pointerEvents: 'none' }}>☰ Layer Controls</h4>
                  {isMobile && <button onClick={() => setLayerControlCollapsed(true)} style={{ border: 'none', background: 'none', color: '#8b949e', cursor: 'pointer', fontSize: '16px' }}>✕</button>}
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', color: '#c9d1d9', fontSize: '14px' }}>
                    <input type="checkbox" checked={showFlood} onChange={e => setShowFlood(e.target.checked)} />
                    <span style={{ width: '12px', height: '12px', backgroundColor: '#ff4d4f', borderRadius: '2px' }}></span> New Flood
                  </label>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', color: '#c9d1d9', fontSize: '14px' }}>
                    <input type="checkbox" checked={showCurrentWater} onChange={e => setShowCurrentWater(e.target.checked)} />
                    <span style={{ width: '12px', height: '12px', backgroundColor: '#1890ff', borderRadius: '2px' }}></span> Current Water
                  </label>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', color: '#c9d1d9', fontSize: '14px' }}>
                    <input type="checkbox" checked={showHistoricWater} onChange={e => setShowHistoricWater(e.target.checked)} />
                    <span style={{ width: '12px', height: '12px', backgroundColor: '#36cfc9', borderRadius: '2px' }}></span> Permanent Water Baseline
                  </label>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid #30363d', paddingTop: '8px' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', color: '#c9d1d9', fontSize: '14px' }}>
                    <input type="checkbox" checked={showActualImage} onChange={e => setShowActualImage(e.target.checked)} />
                    <span style={{ fontSize: '16px' }}>🛰️</span> Actual Satellite Image
                  </label>
                  <button onClick={() => setModalData({ url: meta.latest_rgb, date: meta.latest_date, source: meta.latest_source, title: "Post-Flood Source Image" })} style={{ border: 'none', background: 'none', cursor: 'pointer', fontSize: '16px' }} title="Preview Raw Source">👁️</button>
                </div>
              </div>
            )}

            <MapContainer 
              center={[30.3753, 69.3451]} zoom={DEFAULT_ZOOM} minZoom={DEFAULT_ZOOM} maxZoom={DEFAULT_ZOOM}
              zoomSnap={0.5} zoomControl={false} scrollWheelZoom={false} doubleClickZoom={false} touchZoom={false} dragging={true} 
              style={{ width: '100%', height: '100%' }}
            >
              {/* Always use Satellite Base Map */}
              <TileLayer 
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" 
                attribution="Esri & Carto" 
              />
              <MapController setBounds={setMapBounds} />
              <MapFlyTo targetLocation={targetLocation} />

              {overlayBounds && (
                <>
                  {/* Bottom Layer: Actual Satellite Image */}
                  {showActualImage && meta.latest_rgb && <ImageOverlay url={`${meta.latest_rgb}?t=${Date.now()}`} bounds={overlayBounds} opacity={1.0} />}
                  
                  {/* Layer 2: Permanent Water */}
                  {showHistoricWater && layers.previous && <ImageOverlay url={`${layers.previous}?t=${Date.now()}`} bounds={overlayBounds} opacity={0.6} />}
                  
                  {/* Layer 3: Current Water */}
                  {showCurrentWater && layers.latest && <ImageOverlay url={`${layers.latest}?t=${Date.now()}`} bounds={overlayBounds} opacity={0.6} />}
                  
                  {/* Top Layer: New Flood */}
                  {showFlood && layers.flood && <ImageOverlay url={`${layers.flood}?t=${Date.now()}`} bounds={overlayBounds} opacity={0.8} />}
                </>
              )}
            </MapContainer>
          </div>

        </div>
      </div>
      {modalData && <ImageModal {...modalData} onClose={() => setModalData(null)} />}
    </div>
  );
};

export default App;