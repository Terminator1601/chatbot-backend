/*
* SNAS Progressive Web App
*/
// Check for browser support of service worker
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('service-worker.js', {
		scope: '/'
    })
    .then(function(registration) {
		// Successful registration
		console.log('Hooray. Registration successful, scope is:', registration.scope);
		navigator.serviceWorker.register('service-worker.js', {
		scope: '/'
		});
    }).catch(function(error) {
		// Failed registration, service worker won’t be installed
		console.log('Whoops. Service worker registration failed, error:', error);
    });
}