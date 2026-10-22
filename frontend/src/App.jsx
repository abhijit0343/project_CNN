import React, { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('http://localhost:5000/api/scans/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult(data.data);
      } else {
        setError(data.error || 'Failed to analyze scan.');
      }
    } catch (err) {
      setError('Could not connect to the server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col items-center py-12 px-4">
      <header className="mb-12 text-center">
        <h1 className="text-5xl font-extrabold text-white mb-2 tracking-tight">
          🧠 NeuroScan <span className="text-blue-500">AI</span>
        </h1>
        <p className="text-gray-400 text-lg">Brain Tumor Detection using CNN & MRI Scans</p>
      </header>

      <main className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Upload Section */}
        <section className="bg-gray-900 p-8 rounded-2xl border border-gray-800 shadow-xl">
          <h2 className="text-2xl font-bold mb-6 flex items-center">
            <span className="bg-blue-500/20 text-blue-400 p-2 rounded-lg mr-3">📤</span>
            Upload MRI Scan
          </h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="relative border-2 border-dashed border-gray-700 rounded-xl p-4 hover:border-blue-500/50 transition-colors group">
              <input
                type="file"
                onChange={handleFileChange}
                accept="image/*"
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              {preview ? (
                <img src={preview} alt="Preview" className="w-full h-64 object-cover rounded-lg" />
              ) : (
                <div className="flex flex-col items-center justify-center h-64 text-gray-500 group-hover:text-gray-300">
                  <svg className="w-12 h-12 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
                  </svg>
                  <p>Click or drag to upload MRI image</p>
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={!file || loading}
              className={`w-full py-4 rounded-xl font-bold text-lg transition-all ${
                !file || loading
                  ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20'
              }`}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin h-5 w-5 mr-3 border-2 border-white/20 border-t-white rounded-full" />
                  Analyzing...
                </span>
              ) : 'Analyze Scan'}
            </button>
          </form>
        </section>

        {/* Results Section */}
        <section className="bg-gray-900 p-8 rounded-2xl border border-gray-800 shadow-xl flex flex-col">
          <h2 className="text-2xl font-bold mb-6 flex items-center">
            <span className="bg-green-500/20 text-green-400 p-2 rounded-lg mr-3">📊</span>
            Analysis Result
          </h2>

          {!result && !error && !loading && (
            <div className="flex-1 flex flex-col items-center justify-center text-gray-600">
              <p>Upload an image to see results here</p>
            </div>
          )}

          {error && (
            <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-4 rounded-xl">
              {error}
            </div>
          )}

          {result && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700">
                <p className="text-sm text-gray-400 uppercase tracking-widest font-semibold mb-1">Detected Type</p>
                <h3 className="text-4xl font-black text-white capitalize">{result.prediction}</h3>
                
                <div className="mt-4">
                  <div className="flex justify-between text-sm mb-2">
                    <span>Confidence</span>
                    <span className="font-mono">{(result.confidence * 100).toFixed(2)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 h-3 rounded-full overflow-hidden">
                    <div 
                      className="bg-blue-500 h-full transition-all duration-1000" 
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-sm text-gray-400 uppercase tracking-widest font-semibold mb-4">Class Distribution</h4>
                <div className="space-y-3">
                  {result.top_k?.map((item, idx) => (
                    <div key={idx} className="flex items-center gap-4">
                      <span className="w-24 text-sm truncate capitalize">{item.label}</span>
                      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className="bg-gray-500 h-full" 
                          style={{ width: `${item.confidence * 100}%` }}
                        />
                      </div>
                      <span className="w-12 text-right text-xs font-mono text-gray-500">
                        {(item.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="mt-12 text-gray-500 text-sm">
        <p>&copy; 2026 NeuroScan AI. For research purposes only.</p>
      </footer>
    </div>
  );
}

export default App;
