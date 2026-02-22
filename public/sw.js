// Service Worker for Bilingual Reader - Custom Model Serving
const CACHE_NAME = 'custom-model-cache-v1';

self.addEventListener('install', (event) => {
    self.skipWaiting();
    console.log('[SW] Installed');
});

self.addEventListener('activate', (event) => {
    event.waitUntil(clients.claim());
    console.log('[SW] Activated');
});

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Intercept any request to /custom-model/ path on the same origin
    if (url.pathname.startsWith('/custom-model/')) {
        console.log('[SW] Intercepting:', url.pathname);
        event.respondWith(serveCustomModel(url.pathname));
    }
});

async function serveCustomModel(pathname) {
    try {
        // Extract the filename from /custom-model/path/to/filename.ext
        const relativePath = pathname.replace('/custom-model/', '');
        const filenameOnly = relativePath.split('/').pop() || relativePath;

        console.log('[SW] Looking for:', relativePath, '| filename:', filenameOnly);

        const cache = await caches.open(CACHE_NAME);

        // Try exact match using the stored key format
        let response = await cache.match('/custom-model-files/' + relativePath);

        // Try filename-only match
        if (!response) {
            response = await cache.match('/custom-model-files/' + filenameOnly);
        }

        // Fuzzy fallback for .onnx files (model could be named encoder_model.onnx, etc.)
        if (!response && filenameOnly.endsWith('.onnx')) {
            const keys = await cache.keys();
            const onnxKey = keys.find(k => k.url.endsWith('.onnx'));
            if (onnxKey) {
                console.log('[SW] Fuzzy ONNX match:', onnxKey.url);
                response = await cache.match(onnxKey);
            }
        }

        if (response) {
            console.log('[SW] Serving from cache:', filenameOnly);
            return response.clone();
        }

        // Log all available keys for debugging
        const keys = await cache.keys();
        console.warn('[SW] Not found:', filenameOnly, '| Available:', keys.map(k => k.url.split('/').pop()).join(', '));
        return new Response('File not found: ' + filenameOnly, {
            status: 404,
            headers: { 'Content-Type': 'text/plain' }
        });
    } catch (err) {
        console.error('[SW] Error:', err);
        return new Response('Service Worker error: ' + err.message, { status: 500 });
    }
}
