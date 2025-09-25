// TB Detection System - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // File upload functionality
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');

    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Validate file type
                if (!file.type.match('image.*')) {
                    showAlert('Please select a valid image file (PNG, JPG, JPEG)', 'warning');
                    return;
                }

                // Validate file size (10MB max)
                if (file.size > 10 * 1024 * 1024) {
                    showAlert('File size must be less than 10MB', 'warning');
                    return;
                }

                // Show image preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImg.src = e.target.result;
                    imagePreview.style.display = 'block';
                    
                    // Add smooth animation
                    imagePreview.style.opacity = '0';
                    setTimeout(() => {
                        imagePreview.style.opacity = '1';
                        imagePreview.style.transition = 'opacity 0.3s ease';
                    }, 100);
                };
                reader.readAsDataURL(file);
            }
        });
    }

    // Form submission handler
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const file = fileInput.files[0];
            if (!file) {
                e.preventDefault();
                showAlert('Please select an image file first', 'warning');
                return;
            }

            // Show loading state
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            
            if (loadingSpinner) {
                loadingSpinner.style.display = 'block';
            }
        });
    }

    // Drag and drop functionality
    const uploadArea = document.querySelector('.upload-area');
    if (uploadArea) {
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                fileInput.dispatchEvent(new Event('change'));
            }
        });
    }

    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(alert => {
        setTimeout(() => {
            const closeBtn = alert.querySelector('.btn-close');
            if (closeBtn) {
                closeBtn.click();
            }
        }, 5000);
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Copy to clipboard functionality for API results
    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(function() {
            showAlert('Copied to clipboard!', 'success');
        }).catch(function(err) {
            console.error('Could not copy text: ', err);
        });
    }

    // Progress bar animation
    function animateProgressBar(element, targetValue) {
        const startValue = 0;
        const duration = 1500; // 1.5 seconds
        const startTime = performance.now();

        function updateProgress(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const currentValue = startValue + (targetValue - startValue) * progress;
            element.style.width = currentValue + '%';
            element.setAttribute('aria-valuenow', currentValue);

            if (progress < 1) {
                requestAnimationFrame(updateProgress);
            }
        }

        requestAnimationFrame(updateProgress);
    }

    // Initialize progress bars on results page
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach(bar => {
        const targetValue = parseFloat(bar.getAttribute('data-target-value')) || 0;
        animateProgressBar(bar, targetValue);
    });
});

// Utility function to show alerts
function showAlert(message, type = 'info') {
    const alertsContainer = document.querySelector('.container');
    if (!alertsContainer) return;

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show mt-3`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;

    alertsContainer.insertBefore(alertDiv, alertsContainer.firstChild);

    // Auto-dismiss after 4 seconds
    setTimeout(() => {
        const closeBtn = alertDiv.querySelector('.btn-close');
        if (closeBtn) {
            closeBtn.click();
        }
    }, 4000);
}

// API call function for programmatic access
async function callPredictionAPI(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Dark mode toggle (optional feature)
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
}

// Initialize dark mode from localStorage
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
}

// Add loading animation to buttons
function addButtonLoading(button, loadingText = 'Loading...') {
    const originalText = button.innerHTML;
    button.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${loadingText}`;
    button.disabled = true;
    
    return function() {
        button.innerHTML = originalText;
        button.disabled = false;
    };
}

// Form validation helper
function validateForm(formElement) {
    const inputs = formElement.querySelectorAll('input[required], select[required], textarea[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
            input.classList.add('is-valid');
        }
    });

    return isValid;
}

// Image optimization before upload
function optimizeImage(file, maxWidth = 1024, quality = 0.8) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        img.onload = function() {
            // Calculate new dimensions
            const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
            canvas.width = img.width * ratio;
            canvas.height = img.height * ratio;

            // Draw and compress
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            canvas.toBlob(resolve, 'image/jpeg', quality);
        };

        img.src = URL.createObjectURL(file);
    });
}
