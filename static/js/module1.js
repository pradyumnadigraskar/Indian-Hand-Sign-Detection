 // Fetch and display the video stream
 const videoFeed = document.getElementById('video-feed');
 videoFeed.src = '/video_feed'; // Set the video stream source

 // Clear the sentence
 document.getElementById("clear-btn").addEventListener("click", () => {
     fetch('/clear', { method: 'POST' })
         .then(response => response.json())
         .then(data => {
             document.getElementById("sentence-field").textContent = "";
             document.getElementById("final-field").textContent = "";
         });
 });

 // Speak the sentence
 document.getElementById("speak-btn").addEventListener("click", () => {
     fetch('/speak', { method: 'POST' })
         .then(response => response.json())
         .then(data => {
             console.log("Spoken:", data.status);
         });
 });

 // Periodically fetch updated sentence data
 setInterval(function() {
     fetch('/get_sentence')
         .then(response => response.json())
         .then(data => {
             document.getElementById("sentence-field").textContent = data.sentence;
             document.getElementById("final-field").textContent = data.final_sentence;
         });
 }, 1000); // Update every second