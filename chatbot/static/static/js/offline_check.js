Offline.options = {
    checks: {
        image: {
            url: '/images/offline.png'
        }, 
        active: 'image'
    },
    // Should we check the connection status immediatly on page load.
    checkOnLoad: true,
    // Should we monitor AJAX requests to help decide if we have a connection.
    interceptRequests: true,
    // Should we automatically retest periodically when the connection is down (set to false to disable).
    reconnect: {
        // How many seconds should we wait before rechecking.
        initialDelay: 3,
        // How long should we wait between retries.
        //delay: (1.5 * last delay, capped at 1 hour)
    },
    // Should we store and attempt to remake requests which fail while the connection is down.
    requests: true,
    // Should we show a snake game while the connection is down to keep the user entertained?
    // It's not included in the normal build, you should bring in js/snake.js in addition to
    // offline.min.js.
    game: false
}
window.onload = () => {
    /*
    Offline.check();

    if(Offline.state === "down"){
        document.getElementById("root").classList.toggle('d-none');
        document.getElementById("root").classList.toggle('d-flex');
        document.getElementById("offline_pwa").classList.toggle('d-none');
        document.getElementById("offline_pwa").classList.toggle('d-flex');
    }
    */
}