// DOM Elements
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const charCount = document.getElementById('charCount');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const modelSelect = document.getElementById('modelSelect');
const modelDescription = document.getElementById('modelDescription');

// Result Elements
const sentimentIcon = document.getElementById('sentimentIcon');
const sentimentLabel = document.getElementById('sentimentLabel');
const sentimentBadge = document.getElementById('sentimentBadge');
const originalText = document.getElementById('originalText');
const cleanedText = document.getElementById('cleanedText');
const errorMessage = document.getElementById('errorMessage');
const modelUsed = document.getElementById('modelUsed');

// Available models
let availableModels = {};

// Sentiment Configuration
const sentimentConfig = {
    'positive': {
        icon: '😊',
        class: 'sentiment-positive',
        badgeClass: 'badge-positive'
    },
    'negative': {
        icon: '😢',
        class: 'sentiment-negative',
        badgeClass: 'badge-negative'
    },
    'neutral': {
        icon: '😐',
        class: 'sentiment-neutral',
        badgeClass: 'badge-neutral'
    },
    'sarcastic': {
        icon: '😏',
        class: 'sentiment-warning',
        badgeClass: 'badge-neutral'
    }
};

// Load available models on page load
async function loadAvailableModels() {
    try {
        const response = await fetch('/models');
        availableModels = await response.json();
        
        // Update model select options
        updateModelSelect();
        
        // Set initial description
        updateModelDescription();
        
        console.log('Available models:', availableModels);
    } catch (error) {
        console.error('Failed to load models:', error);
        modelDescription.textContent = 'Using default model';
    }
}

// Update model select dropdown
function updateModelSelect() {
    // Clear existing options
    modelSelect.innerHTML = '';
    
    // Add placeholder option as default
    const placeholderOption = document.createElement('option');
    placeholderOption.value = '';
    placeholderOption.disabled = true;
    placeholderOption.selected = true;
    placeholderOption.textContent = '-- Select Model Version First --';
    modelSelect.appendChild(placeholderOption);
    
    // Add options for available models
    for (const [version, info] of Object.entries(availableModels)) {
        if (info.available) {
            const option = document.createElement('option');
            option.value = version;
            const metadata = info.metadata || {};
            const description = metadata.description || version;
            const architecture = metadata.architecture || '';
            option.textContent = `${version.toUpperCase()} - ${description}`;
            if (architecture) {
                option.textContent += ` (${architecture})`;
            }
            modelSelect.appendChild(option);
        }
    }
}

// Update model description
function updateModelDescription() {
    const selectedVersion = modelSelect.value;
    
    // If no model selected, show instruction
    if (!selectedVersion || selectedVersion === '') {
        modelDescription.textContent = 'Please select a model version to continue';
        return;
    }
    
    const modelInfo = availableModels[selectedVersion];
    
    if (modelInfo && modelInfo.metadata) {
        const metadata = modelInfo.metadata;
        let description = metadata.description || 'No description available';
        
        // Add additional info if available
        if (metadata.epochs) {
            description += ` | ${metadata.epochs} epochs`;
        }
        if (metadata.vocab_size) {
            description += ` | Vocab: ${metadata.vocab_size}`;
        }
        
        modelDescription.textContent = description;
    } else {
        modelDescription.textContent = 'Model information not available';
    }
}

// Model selection change handler
modelSelect.addEventListener('change', () => {
    const selectedValue = modelSelect.value;
    
    // Enable/disable text input and analyze button based on selection
    if (selectedValue && selectedValue !== '') {
        // Model selected - enable inputs
        textInput.disabled = false;
        textInput.placeholder = 'Type or paste your social media text here... 🚀';
        analyzeBtn.disabled = false;
        
        // Focus on text input
        setTimeout(() => textInput.focus(), 100);
    } else {
        // No model selected - disable inputs
        textInput.disabled = true;
        textInput.placeholder = 'Please select a model version first ⬆️';
        textInput.value = '';
        charCount.textContent = '0';
        analyzeBtn.disabled = true;
        hideResults();
        hideError();
    }
    
    updateModelDescription();
    
    // Animate the change
    modelSelect.style.transform = 'scale(1.02)';
    setTimeout(() => {
        modelSelect.style.transform = 'scale(1)';
    }, 200);
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadAvailableModels();
});

// Character Counter
textInput.addEventListener('input', () => {
    const count = textInput.value.length;
    charCount.textContent = count;
    
    // Animate counter
    charCount.style.transform = 'scale(1.2)';
    setTimeout(() => {
        charCount.style.transform = 'scale(1)';
    }, 200);
});

// Clear Button
clearBtn.addEventListener('click', () => {
    textInput.value = '';
    charCount.textContent = '0';
    hideResults();
    hideError();
    
    // Animate clear
    textInput.style.animation = 'fadeOut 0.3s ease';
    setTimeout(() => {
        textInput.style.animation = 'fadeIn 0.3s ease';
    }, 300);
});

// Analyze Button
analyzeBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze!');
        shakeInput();
        return;
    }
    
    await analyzeSentiment(text);
});

// Enter key to analyze
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        analyzeBtn.click();
    }
});

// Analyze Sentiment Function
async function analyzeSentiment(text) {
    try {
        // Show loading state
        showLoading();
        hideResults();
        hideError();
        disableButton();
        
        // Get selected model version
        const modelVersion = modelSelect.value;
        
        // Make API request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                text: text,
                model_version: modelVersion 
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Simulate processing delay for better UX
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError('Oops! Something went wrong. Please try again.');
    } finally {
        hideLoading();
        enableButton();
    }
}

// Display Results
function displayResults(data) {
    const sentiment = data.prediction.toLowerCase();
    const config = sentimentConfig[sentiment] || sentimentConfig['neutral'];
    const confidence = data.confidence || 0;
    const confidencePercent = (confidence * 100).toFixed(2);
    const modelVersion = data.model_version || 'unknown';
    
    // Update sentiment display
    sentimentIcon.textContent = config.icon;
    sentimentLabel.textContent = data.prediction;
    sentimentLabel.className = 'sentiment-label ' + config.class;
    sentimentBadge.className = 'sentiment-badge ' + config.badgeClass;
    
    // Add confidence display
    const confidenceDisplay = document.getElementById('confidenceDisplay');
    if (confidenceDisplay) {
        confidenceDisplay.textContent = `${confidencePercent}%`;
    }
    
    // Display model used
    if (modelUsed) {
        modelUsed.textContent = modelVersion;
    }
    
    // Update text displays
    originalText.textContent = data.input_text;
    cleanedText.textContent = data.cleaned_text;
    
    // Show results with animation
    resultSection.classList.remove('hidden');
    resultSection.style.animation = 'fadeInUp 0.5s ease';
    
    // Animate individual cards
    const cards = document.querySelectorAll('.result-card');
    cards.forEach((card, index) => {
        card.style.animation = 'none';
        setTimeout(() => {
            card.style.animation = `slideInUp 0.5s ease ${index * 0.1}s forwards`;
        }, 10);
    });
    
    // Scroll to results smoothly
    setTimeout(() => {
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
}

// Show Loading
function showLoading() {
    loadingSpinner.classList.remove('hidden');
}

// Hide Loading
function hideLoading() {
    loadingSpinner.classList.add('hidden');
}

// Show Results
function hideResults() {
    resultSection.classList.add('hidden');
}

// Show Error
function showError(message) {
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
    errorSection.style.animation = 'shake 0.5s ease';
}

// Hide Error
function hideError() {
    errorSection.classList.add('hidden');
}

// Disable Button
function disableButton() {
    analyzeBtn.disabled = true;
    analyzeBtn.style.opacity = '0.6';
}

// Enable Button
function enableButton() {
    analyzeBtn.disabled = false;
    analyzeBtn.style.opacity = '1';
}

// Shake Input Animation
function shakeInput() {
    textInput.style.animation = 'shake 0.5s ease';
    setTimeout(() => {
        textInput.style.animation = '';
    }, 500);
}

// Add ripple effect to button
analyzeBtn.addEventListener('click', function(e) {
    const ripple = document.createElement('span');
    ripple.classList.add('ripple-effect');
    this.appendChild(ripple);
    
    const rect = this.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = e.clientX - rect.left - size / 2;
    const y = e.clientY - rect.top - size / 2;
    
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    
    setTimeout(() => {
        ripple.remove();
    }, 600);
});

// Add CSS for ripple effect dynamically
const style = document.createElement('style');
style.textContent = `
    .ripple-effect {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.5);
        transform: scale(0);
        animation: rippleAnimation 0.6s ease-out;
        pointer-events: none;
    }
    
    @keyframes rippleAnimation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
`;
document.head.appendChild(style);

// Welcome message
console.log('%c🎉 Sentiment Slang Analyzer Loaded! ', 'background: #6366f1; color: white; padding: 10px; border-radius: 5px; font-size: 14px;');
console.log('%cPress Ctrl+Enter to analyze text quickly!', 'color: #8b5cf6; font-size: 12px;');
