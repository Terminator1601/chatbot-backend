var CACHE_VERSION = "snas-pwa-cache-v0.0.2";
var urlsToCache = [
  //"/",
  //"manifest.json",
  "favicon.ico",
  "/css/lib/bootstrap.min.css",
  //"/css/styles.css",
  //"/js/pwa.js",
  //"/js/offline_check.js",
  //"/js/lib/bootstrap.bundle.min.js",
  "/js/lib/offline.min.js",
  "/images/snas-logo.png"
];
self.addEventListener("install", function(event) {
  event.waitUntil(
    caches.open(CACHE_VERSION)
      .then(function(cache) {
        // Open a cache and cache our files
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', event => {
	if (event.request.url.match( '^.*(\/images\/offline.png).*$' ) ) {
		event.respondWith(
			caches.match(event.request).then(async function(response) {
				try {
					return await fetch(event.request);
				} catch {
					console.log("Fetch has failed and user is offline");
				}
			})
		);
    }
	else{
		event.respondWith(
			caches.open(CACHE_VERSION).then(cache => {
				
				return cache.match(event.request).then(resp => {
					// Request found in current cache, or fetch the file
					return resp || fetch(event.request).then(response => {
						// Cache the newly fetched file for next time
						cache.put(event.request, response.clone());
						return response;
					}).catch(() => {
						// Fetch has failed and user is offline
						
						// Look in the whole cache to load a fallback version of the file
						return caches.match(event.request).then(fallback => {
							return fallback;
						});
					});
				});
			})
		);
	}
});
/*
self.addEventListener("fetch", function(event) {
  console.log(event.request.url);
  event.respondWith(
      caches.match(event.request).then(function(response) {
          return response || fetch(event.request);
      })
  );
});
*/