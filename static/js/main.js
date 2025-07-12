// Main JavaScript functionality for the Airbnb Price Calculator

document.addEventListener('DOMContentLoaded', function() {
    // Get form elements
    const form = document.getElementById('priceCalculatorForm');
    const submitBtn = document.getElementById('estimateBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoading = submitBtn.querySelector('.btn-loading');
    const fileInput = document.getElementById('datafile');

    // Form validation
    const validateForm = () => {
        const requiredFields = form.querySelectorAll('[required]');
        let isValid = true;
        
        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                isValid = false;
                field.classList.add('is-invalid');
            } else {
                field.classList.remove('is-invalid');
            }
        });
        
        return isValid;
    };

    // File upload validation
    const validateFile = (file) => {
        const maxSize = 16 * 1024 * 1024; // 16MB
        const allowedTypes = ['text/csv', 'application/csv'];
        
        if (!file) {
            return 'Please select a file';
        }
        
        if (file.size > maxSize) {
            return 'File size must be less than 16MB';
        }
        
        if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.csv')) {
            return 'Please select a CSV file';
        }
        
        return null;
    };

    // Show loading state
    const showLoading = () => {
        submitBtn.disabled = true;
        btnText.classList.add('hide');
        btnLoading.classList.remove('d-none');
        btnLoading.classList.add('show');
        form.classList.add('loading');
    };

    // Hide loading state
    const hideLoading = () => {
        submitBtn.disabled = false;
        btnText.classList.remove('hide');
        btnLoading.classList.add('d-none');
        btnLoading.classList.remove('show');
        form.classList.remove('loading');
    };

    // Show error message
    const showError = (message) => {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show';
        alertDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        const container = document.querySelector('.container');
        const firstCard = container.querySelector('.card');
        container.insertBefore(alertDiv, firstCard);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    };

    // File input change handler
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        const errorMessage = validateFile(file);
        
        if (errorMessage) {
            showError(errorMessage);
            fileInput.value = '';
            return;
        }
        
        // Show file info
        const fileInfo = document.createElement('div');
        fileInfo.className = 'alert alert-info mt-2';
        fileInfo.innerHTML = `
            <i class="fas fa-file-csv me-2"></i>
            Selected file: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)
        `;
        
        // Remove existing file info
        const existingInfo = fileInput.parentNode.querySelector('.alert-info');
        if (existingInfo) {
            existingInfo.remove();
        }
        
        fileInput.parentNode.appendChild(fileInfo);
    });

    // Form submission handler
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Validate form
        if (!validateForm()) {
            showError('Please fill in all required fields');
            return;
        }
        
        // Validate file
        const file = fileInput.files[0];
        const fileError = validateFile(file);
        if (fileError) {
            showError(fileError);
            return;
        }
        
        // Show loading state
        showLoading();
        
        // Submit form
        setTimeout(() => {
            form.submit();
        }, 500);
    });

    // Real-time validation for numeric inputs
    const numericInputs = form.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        input.addEventListener('input', function(e) {
            const value = parseFloat(e.target.value);
            const min = parseFloat(e.target.min);
            const max = parseFloat(e.target.max);
            
            if (isNaN(value) || value < min || value > max) {
                e.target.classList.add('is-invalid');
            } else {
                e.target.classList.remove('is-invalid');
            }
        });
    });

    // Smooth scrolling for form sections
    const smoothScroll = (target) => {
        const element = document.querySelector(target);
        if (element) {
            element.scrollIntoView({ 
                behavior: 'smooth',
                block: 'start'
            });
        }
    };

    // Auto-dismiss alerts
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        if (alert.classList.contains('alert-success')) {
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.classList.remove('show');
                    setTimeout(() => alert.remove(), 300);
                }
            }, 10000); // Auto-dismiss success messages after 10 seconds
        }
    });

    // Prevent form resubmission on page refresh
    if (performance.navigation.type === 1) {
        // Page was refreshed
        const currentUrl = new URL(window.location.href);
        if (currentUrl.search) {
            // Remove query parameters and reload
            window.history.replaceState({}, document.title, currentUrl.pathname);
        }
    }

    // Enhanced form interactions
    const formControls = form.querySelectorAll('.form-control');
    formControls.forEach(control => {
        control.addEventListener('focus', function() {
            this.parentNode.classList.add('focused');
        });
        
        control.addEventListener('blur', function() {
            this.parentNode.classList.remove('focused');
        });
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (form.checkValidity()) {
                form.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to clear form
        if (e.key === 'Escape' && e.shiftKey) {
            if (confirm('Are you sure you want to clear the form?')) {
                form.reset();
                // Clear any validation states
                form.querySelectorAll('.is-invalid').forEach(el => {
                    el.classList.remove('is-invalid');
                });
            }
        }
    });

    // Progressive enhancement: Add tooltips to help users
    const addTooltips = () => {
        const tooltipData = {
            'accommodates': 'Maximum number of guests the property can accommodate',
            'bathrooms': 'Number of bathrooms (can be fractional, e.g., 1.5)',
            'bedrooms': 'Number of separate bedrooms',
            'beds': 'Total number of beds available',
            'minimum_nights': 'Minimum number of nights guests must book',
            'maximum_nights': 'Maximum number of nights guests can book',
            'availability_365': 'Number of days the property is available in a year',
            'number_of_reviews': 'Total number of reviews received',
            'reviews_per_month': 'Average number of reviews per month',
            'host_is_superhost': 'Whether the host has achieved Superhost status'
        };
        
        Object.entries(tooltipData).forEach(([id, tooltip]) => {
            const element = document.getElementById(id);
            if (element) {
                element.setAttribute('data-bs-toggle', 'tooltip');
                element.setAttribute('data-bs-placement', 'top');
                element.setAttribute('title', tooltip);
            }
        });
        
        // Initialize Bootstrap tooltips
        if (typeof bootstrap !== 'undefined') {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    };

    // Initialize tooltips
    addTooltips();

    console.log('Airbnb Price Calculator initialized successfully!');
});
