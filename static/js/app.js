// Frontend JavaScript for Explainer Video Generator

// State management
let currentJobId = null;
let pollInterval = null;

// DOM elements
const videoForm = document.getElementById('videoForm');
const promptInput = document.getElementById('prompt');
const titleInput = document.getElementById('title');
const wordCountSpan = document.getElementById('wordCount');
const generateBtn = document.getElementById('generateBtn');
const btnText = generateBtn.querySelector('.btn-text');
const btnLoader = generateBtn.querySelector('.btn-loader');

const statusSection = document.getElementById('statusSection');
const statusMessage = document.getElementById('statusMessage');
const progressFill = document.getElementById('progressFill');

const resultSection = document.getElementById('resultSection');
const videoPlayer = document.getElementById('videoPlayer');
const downloadBtn = document.getElementById('downloadBtn');
const newVideoBtn = document.getElementById('newVideoBtn');

const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const retryBtn = document.getElementById('retryBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Word counter
    promptInput.addEventListener('input', updateWordCount);
    
    // Form submission
    videoForm.addEventListener('submit', handleSubmit);
    
    // Button handlers
    newVideoBtn.addEventListener('click', resetForm);
    retryBtn.addEventListener('click', resetForm);
});

// Update word count
function updateWordCount() {
    const text = promptInput.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    wordCountSpan.textContent = words;
    
    // Visual feedback for word limit
    if (words > 50) {
        wordCountSpan.style.color = '#ef4444';
        wordCountSpan.textContent = `${words} (⚠️ over limit)`;
    } else {
        wordCountSpan.style.color = '#6b7280';
    }
}

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();
    
    const prompt = promptInput.value.trim();
    const title = titleInput.value.trim() || null;
    
    if (!prompt) {
        showError('Please enter a prompt');
        return;
    }
    
    // Check word count
    const wordCount = prompt.split(/\s+/).length;
    if (wordCount > 50) {
        const proceed = confirm(
            `Your prompt has ${wordCount} words (recommended: <50). ` +
            'This may result in longer processing time. Continue?'
        );
        if (!proceed) return;
    }
    
    // Start generation
    await generateVideo(prompt, title);
}

// Generate video
async function generateVideo(prompt, title) {
    try {
        // Show loading state
        setLoading(true);
        hideAllSections();
        showSection(statusSection);
        updateStatus('Submitting request...');
        
        // Submit generation request
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt, title }),
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || data.warning || 'Request failed');
        }
        
        currentJobId = data.job_id;
        updateStatus('Video generation started...');
        
        // Start polling for status
        startPolling();
        
    } catch (error) {
        console.error('Generation error:', error);
        showError(error.message);
        setLoading(false);
    }
}

// Poll for job status
function startPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
    }
    
    // Poll every 2 seconds
    pollInterval = setInterval(checkStatus, 2000);
    
    // Also check immediately
    checkStatus();
}

// Check job status
async function checkStatus() {
    if (!currentJobId) return;
    
    try {
        const response = await fetch(`/api/status/${currentJobId}`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Status check failed');
        }
        
        updateStatus(data.message);
        
        if (data.status === 'completed') {
            stopPolling();
            showVideoResult(data.download_url);
        } else if (data.status === 'failed') {
            stopPolling();
            showError(data.message);
        }
        
    } catch (error) {
        console.error('Status check error:', error);
        stopPolling();
        showError('Failed to check status: ' + error.message);
    }
}

// Stop polling
function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// Show video result
function showVideoResult(downloadUrl) {
    setLoading(false);
    hideAllSections();
    showSection(resultSection);
    
    // Set video source and download link
    videoPlayer.src = downloadUrl;
    downloadBtn.href = downloadUrl;
    
    // Auto-play video (if allowed by browser)
    videoPlayer.play().catch(() => {
        // Auto-play failed, user needs to click play
        console.log('Auto-play not allowed');
    });
}

// Show error
function showError(message) {
    setLoading(false);
    hideAllSections();
    showSection(errorSection);
    errorMessage.textContent = message;
}

// Update status message
function updateStatus(message) {
    statusMessage.textContent = message;
}

// UI helper functions
function setLoading(isLoading) {
    generateBtn.disabled = isLoading;
    
    if (isLoading) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';
    } else {
        btnText.style.display = 'inline-block';
        btnLoader.style.display = 'none';
    }
}

function hideAllSections() {
    statusSection.style.display = 'none';
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
}

function showSection(section) {
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetForm() {
    // Stop any ongoing polling
    stopPolling();
    currentJobId = null;
    
    // Reset form
    videoForm.reset();
    updateWordCount();
    
    // Reset UI
    setLoading(false);
    hideAllSections();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Error handling for fetch
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    // Show error to user
    showError(`Unexpected error: ${event.reason?.message || 'Something went wrong'}`);
});
