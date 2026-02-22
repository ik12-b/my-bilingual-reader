/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI } from "@google/genai";
import * as pdfjsLib from 'pdfjs-dist';
import { Upload, BookOpen, Loader2, Languages, FileText, CheckCircle2, AlertCircle, Eye, EyeOff, X, Columns, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, BarChart3, Clock, Activity, Info, Wifi, WifiOff } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { pipeline, env } from '@xenova/transformers';

// Intercept fetch for custom models
const originalFetch = window.fetch;
(window as any).fetch = async (...args: any[]) => {
  const url = typeof args[0] === 'string' ? args[0] : (args[0] as Request).url;

  // Intercept ALL requests for custom-model files (main thread fallback for Service Worker)
  if (url.includes('/custom-model/')) {
    const parts = url.split('/custom-model/');
    const filenameWithPossiblyPath = parts[parts.length - 1];
    const filenameOnly = filenameWithPossiblyPath.split('/').pop() || '';
    const files = (window as any).__CUSTOM_MODEL_FILES__ as Map<string, Blob>;

    if (files && files.size > 0) {
      // Try exact match first
      let blob = files.get(filenameWithPossiblyPath) || files.get(filenameOnly);

      // Fuzzy fallback for .onnx files
      if (!blob && filenameOnly.endsWith('.onnx')) {
        const anyOnnx = Array.from(files.keys()).find(k => k.endsWith('.onnx'));
        if (anyOnnx) blob = files.get(anyOnnx);
      }

      if (blob) {
        const log = `[Virtual Fetch] ✅ Serving: ${filenameOnly}`;
        console.log(log);
        if ((window as any).__ADD_DEBUG_LOG__) (window as any).__ADD_DEBUG_LOG__(log);
        return new Response(blob, {
          status: 200,
          headers: { 'Content-Type': blob.type || 'application/octet-stream' }
        });
      }

      // IMPORTANT: Return 404 — NOT originalFetch! 
      // originalFetch would hit Vite which returns index.html, crashing Transformers.js JSON parsing.
      const errLog = `[Virtual Fetch] ⚠️ 404: ${filenameOnly} (optional, skipped)`;
      console.warn(errLog);
      if ((window as any).__ADD_DEBUG_LOG__) (window as any).__ADD_DEBUG_LOG__(errLog);
      return new Response(`File not found: ${filenameOnly}`, {
        status: 404,
        headers: { 'Content-Type': 'text/plain' }
      });
    }
  }
  return originalFetch.apply(window, args as [RequestInfo | URL, RequestInit | undefined]);
};

// Configure transformers.js
env.allowLocalModels = true;
env.allowRemoteModels = true;
env.localModelPath = '';
// Note: proxy=false and numThreads=1 keep ONNX on main thread, but
// Service Worker is more reliable as it intercepts ALL fetches.
env.backends.onnx.wasm.proxy = false;
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1/dist/';

// Register Service Worker for custom model serving  
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').then(reg => {
    console.log('[SW] Registered:', reg.scope);
  }).catch(err => {
    console.warn('[SW] Registration failed:', err);
  });
}

// Import worker correctly for Vite
// @ts-ignore - Vite specific import
import pdfWorker from 'pdfjs-dist/build/pdf.worker.mjs?url';

// Try to use local worker, fallback to CDN if needed
try {
  pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;
} catch (e) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;
}

interface TextPair {
  original: string;
  translated: string;
  status: 'pending' | 'translating' | 'completed' | 'error';
  page?: number;
}

interface TranslationPairProps {
  index: number;
  pair: TextPair;
  hoveredIndex: number | null;
  showOriginalOnly: boolean;
  pdfPage: number;
  setHoveredIndex: (index: number | null) => void;
  setPdfPage: (page: number) => void;
  setActiveTab: (tab: 'translation' | 'pdf') => void;
}

const TranslationPair = React.memo(({
  index,
  pair,
  hoveredIndex,
  showOriginalOnly,
  pdfPage,
  setHoveredIndex,
  setPdfPage,
  setActiveTab
}: TranslationPairProps) => {
  return (
    <motion.div
      id={`pair-${index}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className={`group cursor-pointer rounded-xl transition-all duration-300 ${hoveredIndex === index
        ? 'bg-emerald-500/10 -mx-4 px-4 py-2 ring-1 ring-emerald-500/30 scale-[1.01] shadow-lg z-10 relative'
        : 'hover:bg-white/5'
        }`}
      onMouseEnter={() => {
        setHoveredIndex(index);
        if (pair.page && pair.page !== pdfPage) {
          setPdfPage(pair.page);
        }
      }}
      onMouseLeave={() => setHoveredIndex(null)}
      onClick={() => {
        if (pair.page) {
          setPdfPage(pair.page);
          setActiveTab('pdf');
        }
      }}
    >
      <div className="space-y-1">
        <p className="text-xl font-semibold leading-relaxed text-white/90 group-hover:text-emerald-400 transition-colors">
          {pair.original}
        </p>

        {!showOriginalOnly && (
          <div className="min-h-[1.2rem]">
            {pair.status === 'translating' ? (
              <div className="flex items-center gap-2 text-zinc-600 italic text-xs">
                <Loader2 className="w-3 h-3 animate-spin" />
                <span>Menerjemahkan...</span>
              </div>
            ) : pair.status === 'completed' ? (
              <p className="text-base font-medium text-zinc-500 leading-relaxed italic">
                {pair.translated}
              </p>
            ) : pair.status === 'error' ? (
              <p className="text-xs text-red-500/60 italic">Gagal menerjemahkan.</p>
            ) : null}
          </div>
        )}
      </div>
    </motion.div>
  );
});

export default function App() {
  const [textPairs, setTextPairs] = useState<TextPair[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [showOriginalOnly, setShowOriginalOnly] = useState(false);
  const [activeTab, setActiveTab] = useState<'translation' | 'pdf'>('translation');
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfPage, setPdfPage] = useState(1);
  const [numPages, setNumPages] = useState(0);
  const [pdfZoom, setPdfZoom] = useState(1.2);
  const [debouncedZoom, setDebouncedZoom] = useState(1.2);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [textLayerRenderedCount, setTextLayerRenderedCount] = useState(0);
  const [resumedPage, setResumedPage] = useState<number | null>(null);
  const [tokenUsage, setTokenUsage] = useState({
    prompt: 0,
    candidates: 0,
    total: 0
  });
  const [tokenUsageDaily, setTokenUsageDaily] = useState(() => {
    const saved = localStorage.getItem('gemini_daily_usage');
    if (saved) {
      const data = JSON.parse(saved);
      const today = new Date().toISOString().split('T')[0];
      if (data.date === today) return data.total;
    }
    return 0;
  });
  const [showDashboard, setShowDashboard] = useState(false);
  const [isOfflineMode, setIsOfflineMode] = useState(() => {
    const saved = localStorage.getItem('isOfflineMode');
    return saved === 'true';
  });
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [modelProgress, setModelProgress] = useState(0);
  const [modelStatus, setModelStatus] = useState<string | null>(null);
  const [progressItems, setProgressItems] = useState<Record<string, { loaded: number, total: number, progress: number }>>({});
  const [modelSource, setModelSource] = useState<'remote' | 'local' | 'custom'>(() => {
    const saved = localStorage.getItem('modelSource');
    return (saved as 'remote' | 'local' | 'custom') || 'remote';
  });
  const [customModelFiles, setCustomModelFiles] = useState<Map<string, Blob>>(new Map());
  const [customModelName, setCustomModelName] = useState<string>("custom-model");
  const [showPerfDashboard, setShowPerfDashboard] = useState(false);
  const [manualTranslateTrigger, setManualTranslateTrigger] = useState(0);
  const [debugLogs, setDebugLogs] = useState<string[]>([]);

  // Expose log function to fetch interceptor
  useEffect(() => {
    (window as any).__ADD_DEBUG_LOG__ = (msg: string) => {
      setDebugLogs(prev => [msg, ...prev].slice(0, 50));
    };
  }, []);
  const [performanceMetrics, setPerformanceMetrics] = useState({
    avgLatency: 0,
    totalSentences: 0,
    history: [] as { time: number; count: number; latency: number }[],
    lastBatchLatency: 0
  });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textLayerRef = useRef<HTMLDivElement>(null);
  const pdfDocRef = useRef<pdfjsLib.PDFDocumentProxy | null>(null);
  const touchStartDistance = useRef<number | null>(null);
  const initialZoom = useRef<number>(1.2);
  const translatorRef = useRef<any>(null);

  // IndexedDB Persistence for Custom Models
  useEffect(() => {
    const initDB = async () => {
      const request = indexedDB.open('ModelStorage', 1);
      request.onupgradeneeded = (e: any) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains('models')) {
          db.createObjectStore('models');
        }
      };

      request.onsuccess = (e: any) => {
        const db = e.target.result;
        const transaction = db.transaction(['models'], 'readonly');
        const store = transaction.objectStore('models');
        const getRequest = store.get('current_custom_model');

        getRequest.onsuccess = () => {
          if (getRequest.result) {
            const fileMap = new Map<string, Blob>();
            Object.entries(getRequest.result).forEach(([name, blob]: [string, any]) => {
              fileMap.set(name, blob);
            });
            setCustomModelFiles(fileMap);
            console.log("Custom model files loaded from IndexedDB");
          }
        };
      };
    };
    initDB();
  }, []);

  useEffect(() => {
    (window as any).__CUSTOM_MODEL_FILES__ = customModelFiles;

    // Save to Cache API (for Service Worker) and IndexedDB (for persistence)
    if (customModelFiles.size > 0) {
      const saveFiles = async () => {
        try {
          const cache = await caches.open('custom-model-cache-v1');
          for (const [name, blob] of customModelFiles.entries()) {
            await cache.put('/custom-model-files/' + name, new Response(blob));
          }
          console.log('[Cache API] Saved custom model files:', customModelFiles.size);
        } catch (err) {
          console.warn('[Cache API] Failed to save files:', err);
        }
      };
      saveFiles();

      // Also save to IndexedDB for persistence
      const request = indexedDB.open('ModelStorage', 1);
      request.onsuccess = (e: any) => {
        const db = e.target.result;
        const transaction = db.transaction(['models'], 'readwrite');
        const store = transaction.objectStore('models');
        const data: Record<string, Blob> = {};
        customModelFiles.forEach((blob, name) => { data[name] = blob; });
        store.put(data, 'current_custom_model');
      };
    }
  }, [customModelFiles]);

  useEffect(() => {
    localStorage.setItem('isOfflineMode', isOfflineMode.toString());
    localStorage.setItem('modelSource', modelSource);
    if (isOfflineMode && !translatorRef.current && !isModelLoading) {
      loadOfflineModel();
    }
  }, [isOfflineMode, modelSource]);

  const loadOfflineModel = async (overrideFiles?: Map<string, Blob>) => {
    if (translatorRef.current || isModelLoading) return;

    const filesToUse = overrideFiles || customModelFiles;

    // Reset states before starting
    setIsModelLoading(true);
    setProgressItems({});
    setModelProgress(0);
    setErrorMessage(null);

    let modelId = '';
    if (modelSource === 'remote') {
      modelId = 'Xenova/opus-mt-en-id';
      env.localModelPath = ''; // Default
      env.allowRemoteModels = true;
    } else if (modelSource === 'local') {
      modelId = 'opus-mt-en-id';
      env.localModelPath = '/models/';
      env.allowRemoteModels = false; // Forced local
    } else if (modelSource === 'custom') {
      if (filesToUse.size === 0) {
        console.log("Custom model files not ready yet.");
        return;
      }

      // Check if Service Worker is ready - required for custom model loading
      if (!('serviceWorker' in navigator)) {
        throw new Error("Browser tidak mendukung Service Worker. Gunakan Chrome/Edge modern.");
      }

      setModelStatus("Menunggu Service Worker aktif...");

      // Wait for SW to be controlling the page
      const swRegistration = await navigator.serviceWorker.ready;
      const swControlled = navigator.serviceWorker.controller;

      if (!swControlled) {
        // SW just registered but not controlling yet — need page refresh
        setIsModelLoading(false);
        setModelStatus(null);
        setErrorMessage("Service Worker baru aktif! Silakan refresh halaman (Ctrl+R) dan coba lagi.");
        return;
      }

      console.log('[SW] Service Worker is active:', swRegistration.active?.state);

      // Update global reference (for window.fetch fallback)
      (window as any).__CUSTOM_MODEL_FILES__ = filesToUse;

      // CRITICAL: Clear Transformers.js internal cache to avoid stale HTML responses.
      // If Vite previously served index.html for missing files, it gets cached in 'transformers-cache'.
      // This causes JSON parse errors on subsequent load attempts.
      try {
        const cacheNames = await caches.keys();
        const txCaches = cacheNames.filter(n => n.includes('transformers'));
        for (const cacheName of txCaches) {
          const txCache = await caches.open(cacheName);
          const txKeys = await txCache.keys();
          // Delete specific custom-model entries from Transformers cache
          for (const key of txKeys) {
            if (key.url.includes('/custom-model/')) {
              await txCache.delete(key);
              console.log('[Cache Clear] Removed stale entry:', key.url);
            }
          }
        }
        console.log('[Cache Clear] Cleaned Transformers.js cache for custom-model entries');
      } catch (e) {
        console.warn('[Cache Clear] Could not clear Transformers cache:', e);
      }

      // Save to Cache API for Service Worker access
      const cache = await caches.open('custom-model-cache-v1');
      for (const [name, blob] of filesToUse.entries()) {
        await cache.put('/custom-model-files/' + name, new Response(blob, {
          headers: { 'Content-Type': blob.type || 'application/octet-stream' }
        }));
        console.log('[Cache] Saved:', name, blob.size, 'bytes');
      }

      // Transformers.js builds URL as: localModelPath + modelId + '/' + filename
      modelId = 'custom-model';
      env.localModelPath = window.location.origin + '/';
      env.allowLocalModels = true;
      env.allowRemoteModels = false;
    }

    setModelStatus(modelSource === 'remote' ? "Mengunduh model terjemahan..." :
      modelSource === 'local' ? "Memuat model lokal (APK)..." :
        "Memuat model dari penyimpanan browser...");

    try {
      console.log(`[Transformers.js] Initializing pipeline for: ${modelId}`);
      console.log(`[Transformers.js] allowRemoteModels: ${env.allowRemoteModels}, localModelPath: ${env.localModelPath}`);

      const pipelineOptions: any = {
        progress_callback: (data: any) => {
          if (data.status === 'initiate') {
            console.log(`[Transformers.js] Initiating: ${data.file}`);
            const actionLabel = modelSource === 'remote' ? "Mengunduh" : "Memuat";
            setModelStatus(`${actionLabel}: ${data.file}`);
            setProgressItems(prev => ({
              ...prev,
              [data.file]: { loaded: 0, total: 0, progress: 0 }
            }));
          } else if (data.status === 'progress') {
            setProgressItems(prev => {
              const next = { ...prev };
              next[data.file] = {
                loaded: data.loaded || 0,
                total: data.total || 0,
                progress: data.progress || 0
              };

              // Calculate overall progress
              const items = Object.values(next);
              if (items.length === 0) return next;

              const totalProgress = items.reduce((acc, item) => acc + item.progress, 0) / items.length;
              // Ensure it doesn't flicker backwards
              setModelProgress(p => Math.max(p, totalProgress));
              return next;
            });
          } else if (data.status === 'done') {
            console.log(`[Transformers.js] Done: ${data.file}`);
            setModelStatus(`Selesai mengunduh: ${data.file}`);
          } else if (data.status === 'ready') {
            console.log(`[Transformers.js] Ready callback for: ${data.file}`);
          }
        },
        quantized: true,
      };

      const result = await pipeline('translation', modelId, pipelineOptions);

      if (!result) throw new Error("Pipeline returned null or undefined");

      translatorRef.current = result;
      console.log("[Transformers.js] Success: Model loaded and ready");

      setModelProgress(100);
      setProgressItems({});
      setModelStatus("Model siap digunakan.");
      setIsModelLoading(false);
      setTimeout(() => setModelStatus(null), 3000);
    } catch (error: any) {
      console.error("[Transformers.js] Error:", error);
      let errorMsg = error.message || "Model tidak ditemukan atau koneksi terputus.";
      if (modelSource === 'local') {
        errorMsg = "Gagal memuat model lokal. Pastikan folder '/public/models/opus-mt-en-id/' sudah terisi file ONNX.";
      } else if (modelSource === 'remote') {
        errorMsg = "Gagal mengunduh model. Periksa koneksi internet atau coba lagi nanti. Terkadang HuggingFace mengalami gangguan.";
      }
      setErrorMessage(errorMsg);
      setIsOfflineMode(false);
      setIsModelLoading(false);
      setModelStatus(null);
      setProgressItems({});
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedZoom(pdfZoom);
    }, 250);
    return () => clearTimeout(timer);
  }, [pdfZoom]);

  useEffect(() => {
    return () => {
      if (fileUrl) URL.revokeObjectURL(fileUrl);
      if (pdfDocRef.current) {
        pdfDocRef.current.destroy();
        pdfDocRef.current = null;
      }
    };
  }, [fileUrl]);

  useEffect(() => {
    if (fileName && pdfPage > 0) {
      localStorage.setItem(`pdf-progress-${fileName}`, pdfPage.toString());
    }
  }, [fileName, pdfPage]);

  // Sync daily token usage to localStorage
  useEffect(() => {
    const today = new Date().toISOString().split('T')[0];
    localStorage.setItem('gemini_daily_usage', JSON.stringify({
      date: today,
      total: tokenUsageDaily
    }));
  }, [tokenUsageDaily]);

  // PDF Rendering Effect
  useEffect(() => {
    let isMounted = true;
    let renderTask: any = null;

    const renderPdf = async () => {
      if (!fileUrl || !canvasRef.current || activeTab !== 'pdf') return;

      setPdfLoading(true);
      try {
        let pdf = pdfDocRef.current;
        if (!pdf) {
          const loadingTask = pdfjsLib.getDocument(fileUrl);
          pdf = await loadingTask.promise;
          if (!isMounted) {
            pdf.destroy();
            return;
          }
          pdfDocRef.current = pdf;
          setNumPages(pdf.numPages);

          // Ensure pdfPage is within bounds (especially when resuming)
          if (pdfPage > pdf.numPages) {
            setPdfPage(1);
          }
        }

        const page = await pdf.getPage(pdfPage);
        if (!isMounted) return;

        const canvas = canvasRef.current;
        const context = canvas.getContext('2d', { alpha: false });
        if (!context) return;

        // Calculate scale to fit width + user zoom
        const scrollContainer = canvas.closest('.overflow-auto');
        const containerWidth = (scrollContainer?.clientWidth || 800) - 64; // Account for padding
        const unscaledViewport = page.getViewport({ scale: 1 });

        // High DPI Scaling
        // High DPI Scaling - Optimized for sharpness and performance
        const dpr = Math.max(window.devicePixelRatio || 1, 3);
        const baseScale = (containerWidth / unscaledViewport.width) * debouncedZoom;

        // Ensure CSS dimensions are integers to avoid sub-pixel interpolation blur
        const cssWidth = Math.floor(unscaledViewport.width * baseScale);
        const cssHeight = Math.floor(unscaledViewport.height * baseScale);

        canvas.width = cssWidth * dpr;
        canvas.height = cssHeight * dpr;
        canvas.style.width = `${cssWidth}px`;
        canvas.style.height = `${cssHeight}px`;

        // Create a viewport that matches the canvas pixels exactly
        const viewport = page.getViewport({ scale: canvas.width / unscaledViewport.width });

        // Clear canvas
        context.fillStyle = 'white';
        context.fillRect(0, 0, canvas.width, canvas.height);

        // Disable image smoothing for maximum text sharpness (best for vector PDFs)
        context.imageSmoothingEnabled = false;

        const renderContext = {
          canvasContext: context,
          viewport: viewport,
          canvas: canvas,
        };

        renderTask = page.render(renderContext);
        await renderTask.promise;

        // Render Text Layer
        if (textLayerRef.current && isMounted) {
          const textLayerDiv = textLayerRef.current;
          textLayerDiv.innerHTML = '';
          const displayViewport = page.getViewport({ scale: baseScale });
          textLayerDiv.style.height = `${displayViewport.height}px`;
          textLayerDiv.style.width = `${displayViewport.width}px`;

          const textContent = await page.getTextContent();
          if (!isMounted) return;

          // @ts-ignore - TextLayer is a class in PDF.js v4+
          const textLayer = new pdfjsLib.TextLayer({
            textContentSource: textContent,
            container: textLayerDiv,
            viewport: displayViewport
          });
          await textLayer.render();
          setTextLayerRenderedCount(prev => prev + 1);
        }
      } catch (error: any) {
        if (error.name === 'RenderingCancelledException') return;
        console.error("PDF Render error:", error);
      } finally {
        if (isMounted) setPdfLoading(false);
      }
    };

    renderPdf();

    return () => {
      isMounted = false;
      if (renderTask) {
        renderTask.cancel();
      }
    };
  }, [fileUrl, pdfPage, activeTab, debouncedZoom]);

  // PDF-to-List Sync
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      if (!selection || selection.isCollapsed) return;

      const selectedText = selection.toString().trim();
      if (selectedText.length < 10) return;

      const index = textPairs.findIndex(p =>
        p.original.toLowerCase().includes(selectedText.toLowerCase()) ||
        selectedText.toLowerCase().includes(p.original.toLowerCase())
      );

      if (index !== -1) {
        setHoveredIndex(index);
        const element = document.getElementById(`pair-${index}`);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    };

    document.addEventListener('selectionchange', handleSelection);
    return () => document.removeEventListener('selectionchange', handleSelection);
  }, [textPairs]);

  // List-to-PDF Highlight Effect
  useEffect(() => {
    if (hoveredIndex === null || !textLayerRef.current) {
      if (textLayerRef.current) {
        textLayerRef.current.querySelectorAll('span').forEach(s => s.classList.remove('pdf-highlight'));
      }
      return;
    }

    const text = textPairs[hoveredIndex].original.toLowerCase();
    const spans = textLayerRef.current.querySelectorAll('span');
    let firstMatch: HTMLElement | null = null;

    for (const span of Array.from(spans)) {
      const spanText = (span.textContent || '').toLowerCase();
      if (spanText.length > 2 && text.includes(spanText)) {
        span.classList.add('pdf-highlight');
        if (!firstMatch) firstMatch = span as HTMLElement;
      } else {
        span.classList.remove('pdf-highlight');
      }
    }

    if (firstMatch) {
      (firstMatch as HTMLElement).scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [hoveredIndex, pdfPage, pdfZoom, textLayerRenderedCount]);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Clean up old URL and document
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    if (pdfDocRef.current) {
      pdfDocRef.current.destroy();
      pdfDocRef.current = null;
    }

    const url = URL.createObjectURL(file);
    setFileUrl(url);
    setFileName(file.name);
    setIsProcessing(true);
    setProgress(0);
    setTextPairs([]);
    setErrorMessage(null);
    setActiveTab('translation');

    // Load saved progress
    const savedProgress = localStorage.getItem(`pdf-progress-${file.name}`);
    if (savedProgress) {
      const page = parseInt(savedProgress, 10);
      setPdfPage(page);
      setResumedPage(page);
      // Clear resumed message after a few seconds
      setTimeout(() => setResumedPage(null), 5000);
    } else {
      setPdfPage(1);
      setResumedPage(null);
    }

    setPdfZoom(1.2);

    try {
      console.log("Processing file:", file.name, file.type);
      let initialPairs: TextPair[] = [];

      if (file.type === 'application/pdf') {
        initialPairs = await extractSentencesFromPDF(file);
      } else {
        const fullText = await file.text();
        if (!fullText || fullText.trim().length === 0) {
          throw new Error("File kosong atau tidak bisa dibaca.");
        }
        const sentences = splitIntoSentences(fullText);
        initialPairs = sentences.map(s => ({
          original: s,
          translated: '',
          status: 'pending'
        }));
      }

      if (initialPairs.length === 0) {
        throw new Error("Tidak ada teks yang ditemukan dalam file.");
      }

      setTextPairs(initialPairs);
      if (modelSource !== 'custom') {
        await translateInBatches(initialPairs);
      }
    } catch (error: any) {
      console.error("Error processing file:", error);
      setErrorMessage(error.message || "Gagal memproses file.");
    } finally {
      setIsProcessing(false);
    }
  };

  const extractSentencesFromPDF = async (file: File): Promise<TextPair[]> => {
    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;
    pdfDocRef.current = pdf;
    setNumPages(pdf.numPages);

    // Extract text from all pages in parallel
    const pagePromises = Array.from({ length: pdf.numPages }, (_, i) => pdf.getPage(i + 1));
    const pages = await Promise.all(pagePromises);

    const pairs: TextPair[] = [];

    for (let i = 0; i < pages.length; i++) {
      const page = pages[i];
      const content = await page.getTextContent();
      const pageText = content.items.map((item: any) => item.str).join(' ');

      const pageSentences = splitIntoSentences(pageText);
      pageSentences.forEach(s => {
        pairs.push({
          original: s,
          translated: '',
          status: 'pending',
          page: i + 1
        });
      });

      setProgress(Math.round(((i + 1) / pdf.numPages) * 50));
    }
    return pairs;
  };

  const splitIntoSentences = (text: string): string[] => {
    // Faster regex-based splitting
    return text
      .split(/(?<=[.?!])\s+/)
      .map(s => s.trim())
      .filter(s => s.length > 5);
  };

  const translateInBatches = async (pairs: TextPair[]) => {
    const batchSize = 12; // Increased batch size for fewer API calls
    const total = pairs.length;

    const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
    if (!apiKey) {
      setErrorMessage("API Key tidak ditemukan.");
      return;
    }
    const ai = new GoogleGenAI({ apiKey });

    for (let i = 0; i < total; i += batchSize) {
      const currentBatch = pairs.slice(i, i + batchSize);

      setTextPairs(prev => {
        const next = [...prev];
        for (let j = i; j < Math.min(i + batchSize, total); j++) {
          next[j] = { ...next[j], status: 'translating' };
        }
        return next;
      });

      try {
        let translations: string[] = [];

        if (isOfflineMode) {
          if (!translatorRef.current) {
            await loadOfflineModel();
          }

          if (translatorRef.current) {
            const start = performance.now();
            const results = await translatorRef.current(currentBatch.map(p => p.original), {
              max_new_tokens: 256,
            });
            const end = performance.now();
            const latency = end - start;

            translations = results.map((r: any) => r.translation_text);

            // Update performance metrics
            setPerformanceMetrics(prev => {
              const newHistory = [...prev.history, {
                time: Date.now(),
                count: currentBatch.length,
                latency
              }].slice(-50);

              const totalLat = newHistory.reduce((acc, h) => acc + h.latency, 0);
              const totalCount = newHistory.reduce((acc, h) => acc + h.count, 0);

              return {
                avgLatency: totalLat / totalCount,
                totalSentences: prev.totalSentences + currentBatch.length,
                history: newHistory,
                lastBatchLatency: latency
              };
            });
          } else {
            throw new Error("Translator model not ready");
          }
        } else {
          const prompt = `Translate these sentences to Indonesian. Keep it natural. 
          Return ONLY a JSON array of strings.
          Sentences: ${JSON.stringify(currentBatch.map(p => p.original))}`;

          const response = await ai.models.generateContent({
            model: "gemini-1.5-flash-preview-0514",
            contents: [{ parts: [{ text: prompt }] }],
            config: {
              responseMimeType: "application/json"
            }
          });

          translations = JSON.parse(response.text || "[]");

          // Update token usage
          if (response.usageMetadata) {
            const { promptTokenCount = 0, candidatesTokenCount = 0, totalTokenCount = 0 } = response.usageMetadata;
            setTokenUsage(prev => ({
              prompt: prev.prompt + promptTokenCount,
              candidates: prev.candidates + candidatesTokenCount,
              total: prev.total + totalTokenCount
            }));
            setTokenUsageDaily((prev: number) => prev + totalTokenCount);
          }
        }

        setTextPairs(prev => {
          const next = [...prev];
          translations.forEach((t: string, index: number) => {
            const pairIndex = i + index;
            if (pairIndex < total) {
              next[pairIndex] = { ...next[pairIndex], translated: t, status: 'completed' };
            }
          });
          return next;
        });

      } catch (error: any) {
        console.error("Translation error:", error);

        // Handle API Quota Exceeded (429)
        if (error?.status === 429 || (error?.message && error.message.includes('429'))) {
          setErrorMessage("Batas penggunaan API tercapai. Silakan coba lagi nanti.");
        }

        setTextPairs(prev => {
          const next = [...prev];
          for (let j = i; j < Math.min(i + batchSize, total); j++) {
            next[j].status = 'error';
          }
          return next;
        });

        // Stop batching if we hit a quota error to avoid multiple error messages
        if (error?.status === 429) break;
      }

      setProgress(50 + Math.round(((i + batchSize) / total) * 50));
    }
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 2) {
      const distance = Math.hypot(
        e.touches[0].pageX - e.touches[1].pageX,
        e.touches[0].pageY - e.touches[1].pageY
      );
      touchStartDistance.current = distance;
      initialZoom.current = pdfZoom;
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length === 2 && touchStartDistance.current !== null) {
      const distance = Math.hypot(
        e.touches[0].pageX - e.touches[1].pageX,
        e.touches[0].pageY - e.touches[1].pageY
      );
      const ratio = distance / touchStartDistance.current;
      const newZoom = Math.min(5, Math.max(0.5, initialZoom.current * ratio));
      setPdfZoom(newZoom);
    }
  };

  const handleTouchEnd = () => {
    touchStartDistance.current = null;
  };

  return (
    <div className="min-h-screen bg-[#121212] text-white font-sans selection:bg-emerald-500/30">
      <header className="sticky top-0 z-50 bg-[#000000]/80 backdrop-blur-md border-b border-white/10 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-emerald-500 p-2 rounded-lg shadow-lg shadow-emerald-500/20">
              <BookOpen className="w-6 h-6 text-black" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">Bilingual Reader</h1>
              <p className="text-xs text-zinc-400 font-medium uppercase tracking-wider">AI Powered Translation</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {textPairs.length > 0 && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowOriginalOnly(!showOriginalOnly)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-full font-bold transition-all ${showOriginalOnly
                    ? 'bg-emerald-500 text-black'
                    : 'bg-white/10 text-white hover:bg-white/20'
                    }`}
                  title={showOriginalOnly ? "Lihat Bilingual" : "Lihat Teks Asli"}
                >
                  {showOriginalOnly ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  <span className="hidden sm:inline">{showOriginalOnly ? 'Bilingual Off' : 'Teks Asli'}</span>
                </button>

                {fileUrl && fileName?.toLowerCase().endsWith('.pdf') && (
                  <div className="flex items-center gap-1 bg-white/5 rounded-full p-1 border border-white/10">
                    <button
                      onClick={() => setActiveTab('translation')}
                      className={`flex items-center gap-2 px-4 py-1.5 rounded-full font-bold transition-all ${activeTab === 'translation'
                        ? 'bg-emerald-500 text-black'
                        : 'text-white hover:bg-white/10'
                        }`}
                    >
                      <Languages className="w-4 h-4" />
                      <span className="hidden sm:inline">Terjemahan</span>
                    </button>
                    <button
                      onClick={() => setActiveTab('pdf')}
                      className={`flex items-center gap-2 px-4 py-1.5 rounded-full font-bold transition-all ${activeTab === 'pdf'
                        ? 'bg-emerald-500 text-black'
                        : 'text-white hover:bg-white/10'
                        }`}
                    >
                      <FileText className="w-4 h-4" />
                      <span className="hidden sm:inline">Original PDF</span>
                    </button>
                  </div>
                )}
              </div>
            )}

            <button
              onClick={() => setIsOfflineMode(!isOfflineMode)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-full font-bold transition-all ${isOfflineMode
                ? 'bg-amber-500 text-black'
                : 'bg-white/5 text-white hover:bg-white/10'
                }`}
              title={isOfflineMode ? "Seralih ke Online (Gemini)" : "Seralih ke Offline (MarianMT)"}
            >
              {isOfflineMode ? <WifiOff className="w-5 h-5" /> : <Wifi className="w-5 h-5" />}
              <span className="hidden lg:inline">{isOfflineMode ? 'Offline Mode' : 'Online Mode'}</span>
            </button>

            {fileName && (
              <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-full border border-white/10">
                <FileText className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-medium truncate max-w-[150px]">{fileName}</span>
              </div>
            )}

            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isProcessing}
              className="group relative flex items-center gap-2 bg-white text-black px-5 py-2.5 rounded-full font-bold hover:scale-105 transition-all active:scale-95 disabled:opacity-50 disabled:hover:scale-100"
            >
              {isProcessing ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Upload className="w-5 h-5" />
              )}
              <span>{textPairs.length > 0 ? 'Ganti Buku' : 'Pilih Buku'}</span>
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".pdf,.txt"
              className="hidden"
            />
          </div>
        </div>

        {isProcessing && (
          <div className="absolute bottom-0 left-0 w-full h-1 bg-white/5 overflow-hidden">
            <motion.div
              className="h-full bg-emerald-500"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        )}

        {isModelLoading && (
          <div className="absolute bottom-0 left-0 w-full h-1 bg-amber-500/20 overflow-hidden">
            <motion.div
              className="h-full bg-amber-500"
              initial={{ width: 0 }}
              animate={{ width: `${modelProgress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        )}
      </header>

      <main className="relative w-full overflow-hidden">
        {modelStatus && (
          <div className="fixed top-24 left-1/2 -translate-x-1/2 z-[60] w-full max-w-md">
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 bg-[#1a1a1a] border border-amber-500/30 text-white rounded-2xl shadow-2xl overflow-hidden"
            >
              <div className="flex items-center gap-3 mb-3">
                <Loader2 className="w-5 h-5 animate-spin text-amber-500" />
                <div className="flex-1 overflow-hidden">
                  <p className="text-sm font-bold truncate">{modelStatus}</p>
                  {Object.entries(progressItems).some(([_, data]) => data.progress < 100) && (
                    <div className="mt-2 space-y-1.5 max-h-32 overflow-y-auto custom-scrollbar pr-1">
                      {Object.entries(progressItems)
                        .filter(([_, data]) => data.progress < 100)
                        .map(([filename, data]) => (
                          <div key={filename} className="text-[9px] text-zinc-500 flex justify-between items-center bg-white/5 px-2 py-1 rounded-lg border border-white/5">
                            <span className="truncate max-w-[140px] font-mono">{filename}</span>
                            <span className="text-amber-500/80 font-bold">{Math.round(data.progress)}%</span>
                          </div>
                        ))}
                    </div>
                  )}
                </div>
                <span className="ml-auto text-xs font-mono text-amber-500">{Math.round(modelProgress)}%</span>
              </div>
              <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-amber-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${modelProgress}%` }}
                  transition={{ duration: 0.1 }}
                />
              </div>
              <div className="mt-4 flex justify-end">
                <button
                  onClick={() => {
                    setIsModelLoading(false);
                    setIsOfflineMode(false);
                    setModelStatus(null);
                  }}
                  className="text-[10px] text-zinc-500 hover:text-white underline font-bold uppercase tracking-wider"
                >
                  Batalkan / Kembali ke Online
                </button>
              </div>
            </motion.div>
          </div>
        )}

        {errorMessage && (
          <div className="fixed top-24 left-1/2 -translate-x-1/2 z-[60] w-full max-w-md">
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 bg-red-500 text-white rounded-xl shadow-2xl flex items-center gap-3"
            >
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <p className="text-sm font-bold">{errorMessage}</p>
              <button onClick={() => setErrorMessage(null)} className="ml-auto p-1 hover:bg-white/20 rounded">
                <X className="w-4 h-4" />
              </button>
            </motion.div>
          </div>
        )}

        <div className="max-w-7xl mx-auto px-6">
          <AnimatePresence mode="wait">
            {textPairs.length === 0 ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="flex flex-col items-center justify-center py-32 text-center w-full"
              >
                <div className="w-24 h-24 bg-white/5 rounded-full flex items-center justify-center mb-8 border border-white/10">
                  <Languages className="w-12 h-12 text-zinc-500" />
                </div>
                <h2 className="text-3xl font-bold mb-4">Mulai Membaca Bilingual</h2>
                <p className="text-zinc-400 max-w-md text-lg leading-relaxed mb-12">
                  Pilih model penerjemah dan unggah file PDF/TXT Anda untuk mulai membaca dengan terjemahan AI.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-4xl mb-12">
                  {/* Google Gemini (Online) */}
                  <div
                    onClick={() => { setIsOfflineMode(false); setModelSource('remote'); }}
                    className={`p-6 rounded-3xl border transition-all cursor-pointer text-left relative overflow-hidden group ${!isOfflineMode ? 'bg-emerald-500/10 border-emerald-500 shadow-lg shadow-emerald-500/5' : 'bg-white/5 border-white/10 hover:border-white/20'}`}
                  >
                    <div className="flex items-center gap-3 mb-4">
                      <div className={`p-2 rounded-xl ${!isOfflineMode ? 'bg-emerald-500 text-black' : 'bg-white/10 text-zinc-400'}`}>
                        <Wifi className="w-5 h-5" />
                      </div>
                      <span className="font-bold">Gemini AI</span>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Sangat akurat, memerlukan koneksi internet. Menggunakan Gemini 1.5 Flash.
                    </p>
                    {!isOfflineMode && <CheckCircle2 className="absolute top-4 right-4 w-5 h-5 text-emerald-500" />}
                  </div>

                  {/* MarianMT (Remote Download) */}
                  <div
                    onClick={() => { setIsOfflineMode(true); setModelSource('remote'); }}
                    className={`p-6 rounded-3xl border transition-all cursor-pointer text-left relative overflow-hidden group ${isOfflineMode && modelSource === 'remote' ? 'bg-amber-500/10 border-amber-500 shadow-lg shadow-amber-500/5' : 'bg-white/5 border-white/10 hover:border-white/20'}`}
                  >
                    <div className="flex items-center gap-3 mb-4">
                      <div className={`p-2 rounded-xl ${isOfflineMode && modelSource === 'remote' ? 'bg-amber-500 text-black' : 'bg-white/10 text-zinc-400'}`}>
                        <WifiOff className="w-5 h-5" />
                      </div>
                      <span className="font-bold">Offline (Auto)</span>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Lokal di HP, gratis selamanya. Download ~150MB di awal.
                    </p>
                    {isOfflineMode && modelSource === 'remote' && <CheckCircle2 className="absolute top-4 right-4 w-5 h-5 text-amber-500" />}
                  </div>

                  {/* Custom Model (Upload) */}
                  <div
                    onClick={() => { setIsOfflineMode(true); setModelSource('custom'); }}
                    className={`p-6 rounded-3xl border transition-all cursor-pointer text-left relative overflow-hidden group ${isOfflineMode && modelSource === 'custom' ? 'bg-blue-500/10 border-blue-500 shadow-lg shadow-blue-500/5' : 'bg-white/5 border-white/10 hover:border-white/20'}`}
                  >
                    <div className="flex items-center gap-3 mb-4">
                      <div className={`p-2 rounded-xl ${isOfflineMode && modelSource === 'custom' ? 'bg-blue-500 text-black' : 'bg-white/10 text-zinc-400'}`}>
                        <Upload className="w-5 h-5" />
                      </div>
                      <span className="font-bold">Custom</span>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Gunakan file model (.onnx) milik Anda sendiri.
                    </p>
                    {isOfflineMode && modelSource === 'custom' && <CheckCircle2 className="absolute top-4 right-4 w-5 h-5 text-blue-500" />}
                  </div>
                </div>

                {isOfflineMode && modelSource === 'custom' && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-12 p-6 bg-blue-500/5 border border-blue-500/20 rounded-3xl max-w-lg w-full"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h4 className="text-sm font-bold text-blue-400 uppercase tracking-widest">Setup Custom Model</h4>
                        <p className="text-[10px] text-zinc-500 mt-1">Browser akan membaca file langsung dari memori.</p>
                      </div>
                      <button
                        onClick={() => {
                          const input = document.createElement('input');
                          input.type = 'file';
                          input.multiple = true;
                          input.onchange = (e: any) => {
                            const files = e.target.files as FileList;
                            const newMap = new Map<string, Blob>();
                            Array.from(files).forEach(f => newMap.set(f.name, f));
                            setCustomModelFiles(newMap);
                            // Trigger load immediately with the new files to avoid race condition
                            if (isOfflineMode) loadOfflineModel(newMap);
                          };
                          input.click();
                        }}
                        className="text-xs font-bold px-4 py-2 bg-blue-500 text-black rounded-xl hover:bg-blue-400 transition-all shadow-lg shadow-blue-500/20 active:scale-95"
                      >
                        PILIH FILE MODEL
                      </button>
                    </div>

                    {customModelFiles.size > 0 ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-2">
                          {['.onnx', 'config.json', 'tokenizer.json'].map(reqFile => {
                            const found = Array.from(customModelFiles.keys()).some(k => k.toLowerCase().includes(reqFile.toLowerCase()));
                            return (
                              <div key={reqFile} className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-[10px] font-mono ${found ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-500' : 'bg-red-500/10 border-red-500/30 text-red-500'}`}>
                                {found ? <CheckCircle2 className="w-3 h-3" /> : <X className="w-3 h-3" />}
                                {reqFile}
                              </div>
                            );
                          })}
                        </div>

                        <div className="pt-3 border-t border-white/5">
                          <p className="text-[10px] text-zinc-500 mb-2 uppercase tracking-tighter font-bold">Files Uploaded ({customModelFiles.size}):</p>
                          <div className="max-h-24 overflow-y-auto custom-scrollbar flex flex-wrap gap-1">
                            {Array.from(customModelFiles.keys()).map(name => (
                              <span key={name} className="px-2 py-0.5 bg-white/5 rounded text-[9px] text-zinc-400 border border-white/5">
                                {name}
                              </span>
                            ))}
                          </div>
                        </div>

                        {Array.from(customModelFiles.keys()).some(k => k.endsWith('.onnx')) ? (
                          <div className="flex items-center gap-2 text-xs text-emerald-500 bg-emerald-500/20 p-3 rounded-2xl border border-emerald-500/30">
                            <Activity className="w-4 h-4" />
                            <span className="font-bold">Model terdeteksi. Siap untuk dimuat!</span>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 text-xs text-amber-500 bg-amber-500/10 p-3 rounded-2xl border border-amber-500/30">
                            <AlertCircle className="w-4 h-4" />
                            <span>File utama (.onnx) belum ditemukan.</span>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="border-2 border-dashed border-white/10 rounded-2xl py-8 flex flex-col items-center justify-center text-zinc-500 gap-2">
                        <Upload className="w-8 h-8 opacity-20" />
                        <p className="text-xs">Drag & drop atau klik pilih file model</p>
                      </div>
                    )}

                    {debugLogs.length > 0 && (
                      <div className="mt-4 pt-3 border-t border-white/5">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-tighter">Fetch Activity Logs:</p>
                          <button
                            onClick={() => setDebugLogs([])}
                            className="text-[9px] text-zinc-600 hover:text-white underline"
                          >
                            CLEAR LOGS
                          </button>
                        </div>
                        <div className="max-h-32 overflow-y-auto custom-scrollbar bg-black/30 rounded-xl p-2 font-mono text-[9px] space-y-1">
                          {debugLogs.map((log, i) => (
                            <div key={i} className={log.includes('SUCCESS') ? 'text-emerald-500/80' : 'text-amber-500/80'}>
                              {log}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}

                <div className="flex items-center justify-center gap-4">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isProcessing || (isOfflineMode && modelSource === 'custom' && customModelFiles.size === 0)}
                    className="group relative flex items-center gap-3 bg-white text-black px-8 py-4 rounded-full font-bold text-lg hover:scale-105 transition-all shadow-xl shadow-white/10 active:scale-95 disabled:opacity-30 disabled:hover:scale-100"
                  >
                    <Upload className="w-6 h-6" />
                    <span>Pilih Buku (PDF/TXT)</span>
                  </button>
                </div>
              </motion.div>
            ) : (
              <div className="relative min-h-[calc(100vh-120px)] overflow-hidden">
                <motion.div
                  key={activeTab}
                  initial={{ x: activeTab === 'pdf' ? '100%' : '-100%', opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: activeTab === 'pdf' ? '-100%' : '100%', opacity: 0 }}
                  transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                  className="w-full"
                >
                  {activeTab === 'translation' ? (
                    <div className="py-12 max-w-3xl mx-auto space-y-6">
                      {textPairs.map((pair, index) => (
                        <TranslationPair
                          key={index}
                          index={index}
                          pair={pair}
                          hoveredIndex={hoveredIndex}
                          showOriginalOnly={showOriginalOnly}
                          pdfPage={pdfPage}
                          setHoveredIndex={setHoveredIndex}
                          setPdfPage={setPdfPage}
                          setActiveTab={setActiveTab}
                        />
                      ))}

                      {isProcessing && (
                        <div className="flex items-center justify-center py-12">
                          <div className="flex flex-col items-center gap-4">
                            <Loader2 className="w-8 h-8 animate-spin text-emerald-500" />
                            <p className="text-zinc-500 font-medium">Memproses...</p>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="py-8 flex flex-col h-full">
                      <div className="flex items-center justify-center mb-6">
                        <div className="flex items-center gap-4 bg-white/5 px-4 py-1.5 rounded-full border border-white/10">
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => setPdfPage(p => Math.max(1, p - 1))}
                              disabled={pdfPage <= 1}
                              className="p-1 hover:bg-white/10 rounded-lg disabled:opacity-30 transition-colors"
                            >
                              <ChevronLeft className="w-6 h-6" />
                            </button>
                            <span className="text-sm font-mono text-zinc-300 min-w-[80px] text-center">
                              Halaman {pdfPage} / {numPages}
                            </span>
                            <button
                              onClick={() => setPdfPage(p => Math.min(numPages, p + 1))}
                              disabled={pdfPage >= numPages}
                              className="p-1 hover:bg-white/10 rounded-lg disabled:opacity-30 transition-colors"
                            >
                              <ChevronRight className="w-6 h-6" />
                            </button>
                          </div>

                          <div className="w-px h-4 bg-white/20 mx-2" />

                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => setPdfZoom(z => Math.max(0.5, z - 0.25))}
                              className="p-1 hover:bg-white/10 rounded-lg text-zinc-400 transition-colors"
                            >
                              <ZoomOut className="w-5 h-5" />
                            </button>
                            <span className="text-sm font-mono text-zinc-300 min-w-[50px] text-center">
                              {Math.round(pdfZoom * 100)}%
                            </span>
                            <button
                              onClick={() => setPdfZoom(z => Math.min(5, z + 0.25))}
                              className="p-1 hover:bg-white/10 rounded-lg text-zinc-400 transition-colors"
                            >
                              <ZoomIn className="w-5 h-5" />
                            </button>
                          </div>
                        </div>
                      </div>

                      <div
                        className="flex-1 overflow-auto flex justify-center bg-zinc-900 rounded-3xl p-4 md:p-8 border border-white/5 custom-scrollbar min-h-[600px] relative touch-none"
                        onTouchStart={handleTouchStart}
                        onTouchMove={handleTouchMove}
                        onTouchEnd={handleTouchEnd}
                      >
                        {pdfLoading && (
                          <div className="absolute inset-0 z-50 flex items-center justify-center bg-zinc-900/50 backdrop-blur-sm rounded-3xl">
                            <div className="flex flex-col items-center gap-3">
                              <Loader2 className="w-10 h-10 text-emerald-500 animate-spin" />
                              <p className="text-sm font-medium text-zinc-400">Memuat PDF...</p>
                            </div>
                          </div>
                        )}
                        <div className="relative inline-block bg-white shadow-[0_20px_50px_rgba(0,0,0,0.5)] rounded-sm border-[1px] border-white/20 overflow-hidden">
                          <canvas ref={canvasRef} className="block" />
                          <div ref={textLayerRef} className="textLayer" />
                        </div>
                      </div>
                    </div>
                  )}
                </motion.div>
              </div>
            )}
          </AnimatePresence>
        </div>
      </main>

      {textPairs.length > 0 && (
        <div className="fixed bottom-8 right-8 z-50 flex flex-col items-end gap-3">
          <AnimatePresence>
            {resumedPage && (
              <motion.div
                initial={{ opacity: 0, y: 20, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="bg-emerald-500 text-black px-4 py-2 rounded-xl shadow-xl font-bold text-sm flex items-center gap-2"
              >
                <CheckCircle2 className="w-4 h-4" />
                Melanjutkan dari halaman {resumedPage}
              </motion.div>
            )}
          </AnimatePresence>

          <div
            className="bg-[#282828] border border-white/10 rounded-full px-6 py-3 shadow-2xl flex items-center gap-4 cursor-pointer hover:bg-[#333] transition-all hover:scale-[1.02] active:scale-95 group"
            onClick={() => setShowDashboard(true)}
          >
            <div className="flex items-center gap-2">
              <CheckCircle2 className={`w-5 h-5 ${progress === 100 ? 'text-emerald-500' : 'text-zinc-500'}`} />
              <span className="text-sm font-bold">{progress}% Selesai</span>
            </div>
            <div className="w-px h-4 bg-white/10" />
            <div className="flex flex-col">
              <span className="text-xs text-zinc-400 font-mono uppercase tracking-widest group-hover:text-white transition-colors">
                {textPairs.filter(p => p.status === 'completed').length} / {textPairs.length} Kalimat
              </span>
              {tokenUsage.total > 0 && (
                <span className="text-[10px] text-emerald-500/70 font-mono flex items-center gap-1">
                  <Activity className="w-3 h-3" />
                  {tokenUsage.total.toLocaleString()} Tokens
                </span>
              )}
            </div>

            {modelSource === 'custom' && textPairs.some(p => p.status === 'pending') && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  translateInBatches(textPairs);
                }}
                className="ml-2 px-4 py-1.5 bg-blue-500 text-black text-xs font-bold rounded-full hover:bg-blue-400 transition-colors shadow-lg shadow-blue-500/20"
              >
                MULAI TERJEMAHKAN
              </button>
            )}

            <div className="flex gap-2 ml-2">
              <div
                className="p-1.5 bg-emerald-500/10 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity hover:bg-emerald-500/20"
                onClick={(e) => { e.stopPropagation(); setShowDashboard(true); }}
                title="API Monitoring"
              >
                <BarChart3 className="w-4 h-4 text-emerald-500" />
              </div>
              <div
                className="p-1.5 bg-amber-500/10 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity hover:bg-amber-500/20"
                onClick={(e) => { e.stopPropagation(); setShowPerfDashboard(true); }}
                title="Performance Monitoring"
              >
                <Activity className="w-4 h-4 text-amber-500" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* PDF Preview Modal Removed */}

      {/* API Usage Dashboard Modal */}
      <AnimatePresence>
        {showDashboard && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowDashboard(false)}
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="relative w-full max-w-lg bg-[#1a1a1a] border border-white/10 rounded-3xl p-8 shadow-2xl overflow-hidden"
            >
              <div className="absolute top-0 right-0 p-4">
                <button
                  onClick={() => setShowDashboard(false)}
                  className="p-2 hover:bg-white/5 rounded-xl transition-colors text-zinc-400 hover:text-white"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="flex items-center gap-4 mb-8">
                <div className="p-3 bg-emerald-500/10 rounded-2xl">
                  <BarChart3 className="w-6 h-6 text-emerald-500" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">Monitoring API Gemini</h3>
                  <p className="text-sm text-zinc-500">Batas penggunaan & Quota Real-time</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="bg-white/5 border border-white/5 rounded-2xl p-4">
                  <div className="flex items-center gap-2 text-zinc-400 mb-2">
                    <Activity className="w-4 h-4" />
                    <span className="text-xs uppercase font-bold tracking-wider">Harian</span>
                  </div>
                  <div className="text-2xl font-mono font-bold text-emerald-500">
                    {tokenUsageDaily.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-zinc-500 mt-1">Tokens used today</div>
                </div>
                <div className="bg-white/5 border border-white/5 rounded-2xl p-4">
                  <div className="flex items-center gap-2 text-zinc-400 mb-2">
                    <Clock className="w-4 h-4" />
                    <span className="text-xs uppercase font-bold tracking-wider">Reset</span>
                  </div>
                  <div className="text-2xl font-mono font-bold">
                    {new Date().getUTCHours() < 24 ? (23 - new Date().getUTCHours()) : 0}j {60 - new Date().getUTCMinutes()}m
                  </div>
                  <div className="text-[10px] text-zinc-500 mt-1">Estimasi reset (UTC)</div>
                </div>
              </div>

              <div className="space-y-6">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-zinc-400 font-medium">Batas Per Menit (TPM)</span>
                    <span className="text-xs font-mono text-zinc-500 text-right">Limit: 1M Tokens</span>
                  </div>
                  <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min((tokenUsage.total / 1000000) * 100, 100)}%` }}
                      className="h-full bg-emerald-500"
                    />
                  </div>
                </div>

                <div className="bg-[#222] rounded-2xl p-4 text-xs leading-relaxed text-zinc-400 border border-white/5">
                  <p className="flex items-start gap-2">
                    <Info className="w-4 h-4 text-zinc-500 shrink-0 mt-0.5" />
                    <span>
                      Data ini adalah estimasi berdasarkan penggunaan aplikasi. Google Gemini Free Tier memiliki limit 15 RPM dan 1 Juta TPM. Quota harian di-reset setiap hari pukul 00:00 UTC.
                    </span>
                  </p>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Performance Monitoring Dashboard Modal */}
      <AnimatePresence>
        {showPerfDashboard && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowPerfDashboard(false)}
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="relative w-full max-w-lg bg-[#1a1a1a] border border-white/10 rounded-3xl p-8 shadow-2xl overflow-hidden"
            >
              <div className="absolute top-0 right-0 p-4">
                <button
                  onClick={() => setShowPerfDashboard(false)}
                  className="p-2 hover:bg-white/5 rounded-xl transition-colors text-zinc-400 hover:text-white"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="flex items-center gap-4 mb-8">
                <div className="p-3 bg-amber-500/10 rounded-2xl">
                  <Activity className="w-6 h-6 text-amber-500" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">Peforma Model Offline</h3>
                  <p className="text-sm text-zinc-500">Latency & throughput monitoring</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="bg-white/5 border border-white/5 rounded-2xl p-4">
                  <div className="flex items-center gap-2 text-zinc-400 mb-2">
                    <Activity className="w-3 h-3 text-amber-500" />
                    <span className="text-xs uppercase font-bold tracking-wider">Avg Latency</span>
                  </div>
                  <div className="text-2xl font-mono font-bold text-amber-500">
                    {performanceMetrics.avgLatency ? Math.round(performanceMetrics.avgLatency) : '-'}
                  </div>
                  <div className="text-[10px] text-zinc-500 mt-1">ms per sentence</div>
                </div>
                <div className="bg-white/5 border border-white/5 rounded-2xl p-4">
                  <div className="flex items-center gap-2 text-zinc-400 mb-2">
                    <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                    <span className="text-xs uppercase font-bold tracking-wider">Total</span>
                  </div>
                  <div className="text-2xl font-mono font-bold">
                    {performanceMetrics.totalSentences.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-zinc-500 mt-1">Sentences translated</div>
                </div>
              </div>

              <div className="space-y-6">
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-sm text-zinc-400 font-bold uppercase tracking-widest">Model Configuration</span>
                    <span className="px-2 py-0.5 bg-amber-500/10 text-amber-500 text-[10px] font-bold rounded uppercase">
                      Quantized 8-bit
                    </span>
                  </div>
                  <div className="bg-[#222] rounded-2xl p-6 border border-white/5 space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Model Source</span>
                      <select
                        value={modelSource}
                        onChange={(e) => {
                          const newSource = e.target.value as 'remote' | 'local';
                          setModelSource(newSource);
                          translatorRef.current = null; // Clear to force reload
                          if (isOfflineMode) loadOfflineModel();
                        }}
                        className="bg-[#333] border border-white/10 rounded-lg px-3 py-1.5 text-xs font-bold outline-none focus:ring-1 ring-amber-500 transition-all cursor-pointer"
                      >
                        <option value="remote">Remote (HuggingFace CDN)</option>
                        <option value="local">Local (Bundled in APK)</option>
                        <option value="custom">Custom (Upload Model)</option>
                      </select>
                    </div>
                    <div className="flex items-center justify-between pt-2 border-t border-white/5">
                      <span className="text-zinc-500 text-xs">Model ID</span>
                      <span className="text-xs font-mono text-zinc-400">
                        {modelSource === 'local' ? 'opus-mt-en-id' : modelSource === 'custom' ? 'User-Uploaded Model' : 'Xenova/opus-mt-en-id'}
                      </span>
                    </div>

                    {modelSource === 'custom' && (
                      <div className="pt-2 flex flex-col gap-2">
                        <div className="flex items-center justify-between text-[10px] text-zinc-500 font-mono">
                          <span>Files: {customModelFiles.size} detected</span>
                          <button
                            onClick={() => {
                              const input = document.createElement('input');
                              input.type = 'file';
                              input.multiple = true;
                              input.onchange = (e: any) => {
                                const files = e.target.files as FileList;
                                const newMap = new Map<string, Blob>();
                                Array.from(files).forEach(f => {
                                  newMap.set(f.name, f);
                                });
                                setCustomModelFiles(newMap);
                                if (isOfflineMode) loadOfflineModel();
                              };
                              input.click();
                            }}
                            className="text-amber-500 hover:text-amber-400 font-bold"
                          >
                            CHANGE FILES
                          </button>
                          <button
                            onClick={() => {
                              if (confirm("Hapus model custom dari penyimpanan browser?")) {
                                setCustomModelFiles(new Map());
                                const request = indexedDB.open('ModelStorage', 1);
                                request.onsuccess = (e: any) => {
                                  const db = e.target.result;
                                  const transaction = db.transaction(['models'], 'readwrite');
                                  const store = transaction.objectStore('models');
                                  store.delete('current_custom_model');
                                };
                                setModelSource('remote');
                                translatorRef.current = null;
                              }
                            }}
                            className="text-zinc-600 hover:text-red-500 font-bold ml-3"
                          >
                            CLEAR
                          </button>
                        </div>
                        {customModelFiles.size > 0 && Array.from(customModelFiles.keys()).some(k => k.endsWith('.onnx')) ? (
                          <div className="text-[9px] text-emerald-500 bg-emerald-500/10 px-2 py-1 rounded inline-flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3" />
                            Model file detected (.onnx)
                          </div>
                        ) : customModelFiles.size > 0 ? (
                          <div className="text-[9px] text-red-500 bg-red-500/10 px-2 py-1 rounded inline-flex items-center gap-1">
                            <AlertCircle className="w-3 h-3" />
                            Missing .onnx file!
                          </div>
                        ) : null}
                      </div>
                    )}

                    <div className="flex items-center justify-start gap-2 pt-2 text-[10px] text-zinc-500">
                      <Info className="w-3 h-3" />
                      <span>{modelSource === 'local' ? 'Local mode memerlukan model di folder `/public/models/`.' :
                        modelSource === 'custom' ? 'Upload semua file model (onnx, json) sekaligus.' :
                          'Mengunduh dari CDN HuggingFace.'}</span>
                    </div>
                  </div>
                </div>

                {performanceMetrics.history.length > 0 && (
                  <div>
                    <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-widest mb-3">Terakhir (20 Batch)</h3>
                    <div className="h-20 flex items-end gap-1.5 px-1 bg-white/5 rounded-2xl p-4 border border-white/5">
                      {performanceMetrics.history.slice(-20).map((h, i) => (
                        <div
                          key={i}
                          className="flex-1 bg-amber-500/30 rounded-t-sm hover:bg-amber-500 transition-all group relative cursor-help"
                          style={{ height: `${Math.min(100, (h.latency / h.count) / 4)}%` }} // 4ms per sentence = 1%
                        >
                          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-white text-black text-[10px] font-bold rounded opacity-0 group-hover:opacity-100 whitespace-nowrap z-10 shadow-xl">
                            {Math.round(h.latency / h.count)}ms
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
