// Power Usage Forecasting Tool - Main JavaScript

// Global variables
let currentUsage = 0;
let updateInterval;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Power Usage Forecasting Tool initialized');
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
    
    // Initialize real-time updates if on dashboard
    if (window.location.pathname === '/dashboard') {
        initializeRealTimeUpdates();
    }
});

// Real-time updates functionality
function initializeRealTimeUpdates() {
    // Update every 30 seconds
    updateInterval = setInterval(updateCurrentUsage, 30000);
    
    // Initial update
    updateCurrentUsage();
}

// Update current usage display
function updateCurrentUsage() {
    fetch('/api/current_usage')
        .then(response => response.json())
        .then(data => {
            currentUsage = data.usage;
            
            // Update usage displays
            const usageElements = document.querySelectorAll('.current-usage');
            usageElements.forEach(element => {
                element.textContent = `${currentUsage.toFixed(2)} kWh`;
            });
            
            // Update usage indicators
            updateUsageIndicators(currentUsage);
            
            // Update timestamp
            const timestampElements = document.querySelectorAll('.last-updated');
            const timestamp = new Date(data.timestamp).toLocaleString();
            timestampElements.forEach(element => {
                element.textContent = `Last updated: ${timestamp}`;
            });
        })
        .catch(error => {
            console.error('Error updating current usage:', error);
        });
}

// Update usage indicators based on current usage
function updateUsageIndicators(usage) {
    const indicators = document.querySelectorAll('.usage-indicator');
    const threshold = 4.0; // kWh threshold for high usage
    
    indicators.forEach(indicator => {
        indicator.classList.remove('usage-normal', 'usage-warning', 'usage-critical');
        
        if (usage > threshold * 1.5) {
            indicator.classList.add('usage-critical');
        } else if (usage > threshold) {
            indicator.classList.add('usage-warning');
        } else {
            indicator.classList.add('usage-normal');
        }
    });
}

// Format numbers for display
function formatNumber(number, decimals = 2) {
    return Number(number).toFixed(decimals);
}

// Format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

// Format percentage
function formatPercentage(value) {
    return `${(value * 100).toFixed(1)}%`;
}

// Show notification
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the main container
    const container = document.querySelector('main');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Show loading spinner
function showLoading(element) {
    const spinner = document.createElement('div');
    spinner.className = 'spinner-border text-primary';
    spinner.setAttribute('role', 'status');
    spinner.innerHTML = '<span class="visually-hidden">Loading...</span>';
    
    element.appendChild(spinner);
    element.style.opacity = '0.6';
    element.style.pointerEvents = 'none';
}

// Hide loading spinner
function hideLoading(element) {
    const spinner = element.querySelector('.spinner-border');
    if (spinner) {
        spinner.remove();
    }
    element.style.opacity = '1';
    element.style.pointerEvents = 'auto';
}

// Validate form inputs
function validateForm(formElement) {
    const inputs = formElement.querySelectorAll('input[required], select[required], textarea[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
        }
    });
    
    return isValid;
}

// Handle form submission
function handleFormSubmit(formElement, successCallback = null) {
    formElement.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!validateForm(formElement)) {
            showNotification('Please fill in all required fields.', 'warning');
            return;
        }
        
        const submitButton = formElement.querySelector('button[type="submit"]');
        const originalText = submitButton.innerHTML;
        
        showLoading(submitButton);
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
        
        // Submit form data
        const formData = new FormData(formElement);
        
        fetch(formElement.action, {
            method: formElement.method,
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading(submitButton);
            submitButton.innerHTML = originalText;
            
            if (data.success) {
                showNotification(data.message || 'Operation completed successfully!', 'success');
                if (successCallback) {
                    successCallback(data);
                }
            } else {
                showNotification(data.message || 'An error occurred.', 'danger');
            }
        })
        .catch(error => {
            hideLoading(submitButton);
            submitButton.innerHTML = originalText;
            showNotification('An error occurred while processing your request.', 'danger');
            console.error('Form submission error:', error);
        });
    });
}

// Export data functionality
function exportData(startDate = null, endDate = null) {
    if (!startDate) {
        startDate = new Date();
        startDate.setDate(startDate.getDate() - 30);
    }
    if (!endDate) {
        endDate = new Date();
    }
    
    const url = `/api/export_data?start_date=${startDate.toISOString().split('T')[0]}&end_date=${endDate.toISOString().split('T')[0]}`;
    
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error('Export failed');
            }
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `power_usage_${new Date().toISOString().split('T')[0]}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
            
            showNotification('Data exported successfully!', 'success');
        })
        .catch(error => {
            console.error('Export error:', error);
            showNotification('Error exporting data. Please try again.', 'danger');
        });
}

// Generate sample data
function generateSampleData() {
    const button = event.target;
    const originalText = button.innerHTML;
    
    showLoading(button);
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Generating...';
    
    fetch('/api/generate_sample_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        hideLoading(button);
        button.innerHTML = originalText;
        
        if (data.success) {
            showNotification('Sample data generated successfully!', 'success');
            setTimeout(() => {
                location.reload();
            }, 1000);
        } else {
            showNotification('Error generating sample data: ' + data.message, 'danger');
        }
    })
    .catch(error => {
        hideLoading(button);
        button.innerHTML = originalText;
        showNotification('Error generating sample data', 'danger');
        console.error('Error:', error);
    });
}

// Chart utilities
function createChart(canvasId, type, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) {
        console.error(`Canvas with id '${canvasId}' not found`);
        return null;
    }
    
    return new Chart(ctx, {
        type: type,
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            ...options
        }
    });
}

// Date utilities
function formatDate(date) {
    return new Date(date).toLocaleDateString();
}

function formatDateTime(date) {
    return new Date(date).toLocaleString();
}

function getRelativeTime(date) {
    const now = new Date();
    const diff = now - new Date(date);
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    if (hours < 24) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    return `${days} day${days > 1 ? 's' : ''} ago`;
}

// Utility functions for data processing
function calculateAverage(data) {
    if (data.length === 0) return 0;
    return data.reduce((sum, value) => sum + value, 0) / data.length;
}

function calculateTotal(data) {
    return data.reduce((sum, value) => sum + value, 0);
}

function findMax(data) {
    return Math.max(...data);
}

function findMin(data) {
    return Math.min(...data);
}

// Cleanup function
function cleanup() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
}

// Handle page unload
window.addEventListener('beforeunload', cleanup);

// Export functions for use in other scripts
window.PowerForecast = {
    formatNumber,
    formatCurrency,
    formatPercentage,
    showNotification,
    showLoading,
    hideLoading,
    validateForm,
    handleFormSubmit,
    exportData,
    generateSampleData,
    createChart,
    formatDate,
    formatDateTime,
    getRelativeTime,
    calculateAverage,
    calculateTotal,
    findMax,
    findMin,
    updateCurrentUsage,
    updateUsageIndicators
}; 