// assets/fade.js

function applyFadeIn() {
    var content = document.getElementById("page-content");
    console.log("applyFadeIn called, page-content:", content);  // Debug log
    if (content) {
        // Remove the class first
        content.classList.remove("loaded");
        // Add it back after a delay to trigger transition
        setTimeout(function () {
            content.classList.add("loaded");
            console.log("added 'loaded' class to page-content");
        }, 500);  // 500ms delay for testing
    } else {
        console.log("No element with id 'page-content' found");
    }
}

// Set up the MutationObserver to re-apply fade whenever the content changes
var targetNode = document.getElementById("page-content");
if (targetNode) {
    var observer = new MutationObserver(function(mutations) {
        console.log("MutationObserver triggered");
        applyFadeIn();
    });
    observer.observe(targetNode, {childList: true, subtree: true});
} else {
    console.log("No 'page-content' element available for observation");
}

// Also run on initial load
document.addEventListener("DOMContentLoaded", function() {
    console.log("DOMContentLoaded event fired");
    applyFadeIn();
});
