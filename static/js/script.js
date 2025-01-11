// Array of background images
const backgrounds = [
    'background_1.jpg',
    'background_2.jpg',
    'background_3.jpg',
    'background_4.jpg'
];

// Function to change background images
let currentIndex = 0;

function changeBackground() {
    const backgroundAnimation = document.querySelector('.background-animation');
    backgroundAnimation.style.backgroundImage = `url(${backgrounds[currentIndex]})`;
    currentIndex = (currentIndex + 1) % backgrounds.length;
}

// Change background every 5 seconds
setInterval(changeBackground, 5000);

// Function to navigate to options (you can modify this as needed)
function navigateToOptions() {
    // For demonstration, it will just alert the user
    alert("Navigating to options...");
    // You can redirect to another page or section as needed
    // window.location.href = "options.html"; // Example of navigation
}

// Initial background setup
changeBackground();