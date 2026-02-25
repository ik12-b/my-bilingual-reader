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
    if (url.pathname.includes('/custom-model/')) {
        console.log('[SW] Intercepting:', url.pathname);
        event.respondWith(serveCustomModel(url.pathname));
    }
});

async function serveCustomModel(pathname) {
    try {
        // Strip query strings if any (though URL pathname usually doesn't include them, 
        // Transformers.js might sometimes append them in ways that affect matching)
        const cleanPathname = pathname.split('?')[0];

        // Extract the relative path after /custom-model/
        const parts = cleanPathname.split('/custom-model/');
        const relativePath = parts[parts.length - 1];
        const filenameOnly = relativePath.split('/').pop() || relativePath;

        console.log('[SW] Looking for:', relativePath, '| filename:', filenameOnly);

        const cache = await caches.open(CACHE_NAME);

        // 1. Exact match
        let response = await cache.match('/custom-model-files/' + relativePath);

        // 2. Filename-only match
        if (!response) {
            response = await cache.match('/custom-model-files/' + filenameOnly);
        }

        // 3. Advanced Fuzzy fallback for .onnx files
        if (!response && filenameOnly.endsWith('.onnx')) {
            const keys = await cache.keys();
            const filenames = keys.map(k => k.url.split('/').pop());

            const isEncoder = filenameOnly.toLowerCase().includes('encoder');
            const isDecoder = filenameOnly.toLowerCase().includes('decoder');

            if (isEncoder) {
                const match = keys.find(k => k.url.toLowerCase().includes('encoder') && k.url.endsWith('.onnx'));
                if (match) response = await cache.match(match);
            } else if (isDecoder || filenameOnly.toLowerCase().includes('merged')) {
                // Try to find a file containing 'decoder' or 'merged'
                const match = keys.find(k => (k.url.toLowerCase().includes('decoder') || k.url.toLowerCase().includes('merged')) && k.url.endsWith('.onnx'));
                if (match) response = await cache.match(match);
            }

            // Fallback to any ONNX if only one exists
            if (!response) {
                const onnxKeys = keys.filter(k => k.url.endsWith('.onnx'));
                if (onnxKeys.length === 1) {
                    console.log('[SW] Single ONNX fallback:', onnxKeys[0].url);
                    response = await cache.match(onnxKeys[0]);
                }
            }
        }

        if (response) {
            console.log('[SW] ✅ Served dari cache:', filenameOnly);
            const newHeaders = new Headers(response.headers);
            newHeaders.set('Access-Control-Allow-Origin', '*');

            return new Response(response.body, {
                status: response.status,
                statusText: response.statusText,
                headers: newHeaders
            });
        }

        // JIKA FILE TIDAK ADA DI CACHE - JANGAN UNDUH DARI INTERNET (Mode Custom)
        console.warn('[SW] ❌ File tidak ditemukan di cache lokal:', filenameOnly);
        return new Response('File kustom tidak ditemukan di cache: ' + filenameOnly, {
            status: 404,
            headers: { 'Content-Type': 'text/plain' }
        });

    } catch (fetchErr) {
        console.error('[SW] ❌ Gagal mengunduh dari network:', fetchErr);
    }
};        