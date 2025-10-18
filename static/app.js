// Enhanced JavaScript functionality for Customer Churn Prediction App

class ChurnPredictor {
    constructor() {
        this.initializeApp();
        this.setupEventListeners();
        this.setupFormValidation();
    }

    initializeApp() {
        console.log('🚀 Churn Predictor App initialized');
        this.showWelcomeMessage();
        this.setupProgressIndicator();
    }

    showWelcomeMessage() {
        // Add a subtle welcome animation
        const mainContainer = document.querySelector('.main-container');
        mainContainer.style.opacity = '0';
        mainContainer.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            mainContainer.style.transition = 'all 0.8s ease';
            mainContainer.style.opacity = '1';
            mainContainer.style.transform = 'translateY(0)';
        }, 100);
    }

    setupProgressIndicator() {
        const progressDiv = document.createElement('div');
        progressDiv.className = 'progress-indicator';
        document.querySelector('.card-body').prepend(progressDiv);
    }

    setupEventListeners() {
        // Form input change listeners
        const formInputs = document.querySelectorAll('#predictionForm input, #predictionForm select');
        formInputs.forEach(input => {
            input.addEventListener('change', () => this.updateProgress());
            input.addEventListener('input', () => this.validateField(input));
        });

        // Enhanced form submission
        document.getElementById('predictionForm').addEventListener('submit', (e) => {
            this.handleFormSubmission(e);
        });

        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
            }
        });
    }

    setupFormValidation() {
        const validationRules = {
            age: { min: 18, max: 100, message: 'Age must be between 18 and 100' },
            creditScore: { min: 300, max: 850, message: 'Credit score must be between 300 and 850' },
            tenure: { min: 0, max: 50, message: 'Tenure must be between 0 and 50 years' },
            balance: { min: 0, message: 'Balance cannot be negative' },
            estimatedSalary: { min: 0, message: 'Salary cannot be negative' }
        };

        Object.keys(validationRules).forEach(fieldName => {
            const field = document.getElementById(fieldName);
            if (field) {
                field.addEventListener('blur', () => this.validateField(field, validationRules[fieldName]));
            }
        });
    }

    validateField(field, rules = null) {
        const value = parseFloat(field.value);
        let isValid = true;
        let message = '';

        if (rules) {
            if (rules.min !== undefined && value < rules.min) {
                isValid = false;
                message = rules.message;
            }
            if (rules.max !== undefined && value > rules.max) {
                isValid = false;
                message = rules.message;
            }
        }

        // Visual feedback
        if (isValid) {
            field.classList.remove('is-invalid');
            field.classList.add('is-valid');
            this.removeFieldError(field);
        } else {
            field.classList.remove('is-valid');
            field.classList.add('is-invalid');
            this.showFieldError(field, message);
        }

        return isValid;
    }

    showFieldError(field, message) {
        this.removeFieldError(field);
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'invalid-feedback';
        errorDiv.textContent = message;
        field.parentNode.appendChild(errorDiv);
    }

    removeFieldError(field) {
        const existingError = field.parentNode.querySelector('.invalid-feedback');
        if (existingError) {
            existingError.remove();
        }
    }

    updateProgress() {
        const formInputs = document.querySelectorAll('#predictionForm input[required], #predictionForm select[required]');
        const checkboxes = document.querySelectorAll('#predictionForm input[type="checkbox"]');
        
        let filledFields = 0;
        const totalFields = formInputs.length;

        formInputs.forEach(input => {
            if (input.value.trim() !== '') {
                filledFields++;
            }
        });

        const progress = (filledFields / totalFields) * 100;
        const progressBar = document.querySelector('.progress-indicator');
        
        if (progressBar) {
            progressBar.style.transform = `scaleX(${progress / 100})`;
            if (progress > 50) {
                progressBar.classList.add('active');
            }
        }
    }

    async handleFormSubmission(e) {
        e.preventDefault();
        
        // Validate all fields before submission
        const formInputs = document.querySelectorAll('#predictionForm input, #predictionForm select');
        let allValid = true;
        
        formInputs.forEach(input => {
            if (!this.validateField(input)) {
                allValid = false;
            }
        });

        if (!allValid) {
            this.showNotification('Please fix validation errors before submitting', 'error');
            return;
        }

        await this.submitPrediction();
    }

    async submitPrediction() {
        const loadingSpinner = document.querySelector('.loading');
        const submitBtn = document.querySelector('.btn-predict');
        const originalText = submitBtn.innerHTML;
        
        // Enhanced loading state
        loadingSpinner.classList.add('show');
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing Customer Data...';

        // Add progress animation
        this.animateLoadingProgress();

        // Collect form data
        const formData = new FormData(document.getElementById('predictionForm'));
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }

        // Handle checkboxes
        data.HasCrCard = document.getElementById('hasCrCard').checked ? 1 : 0;
        data.IsActiveMember = document.getElementById('isActiveMember').checked ? 1 : 0;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                this.displayResults(result);
                this.showNotification('Prediction completed successfully!', 'success');
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
        } catch (error) {
            this.showNotification('Error: ' + error.message, 'error');
            console.error('Prediction error:', error);
        } finally {
            // Reset button state
            loadingSpinner.classList.remove('show');
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    }

    animateLoadingProgress() {
        const messages = [
            'Processing customer data...',
            'Applying neural network model...',
            'Calculating churn probability...',
            'Generating insights...'
        ];

        let messageIndex = 0;
        const submitBtn = document.querySelector('.btn-predict');
        
        const messageInterval = setInterval(() => {
            if (messageIndex < messages.length && submitBtn.disabled) {
                submitBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>${messages[messageIndex]}`;
                messageIndex++;
            } else {
                clearInterval(messageInterval);
            }
        }, 1000);
    }

    displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        const resultCard = document.getElementById('resultCard');
        const resultHeader = document.getElementById('resultHeader');
        const probabilityCircle = document.getElementById('probabilityCircle');
        const probabilityText = document.getElementById('probabilityText');
        const predictionResult = document.getElementById('predictionResult');
        const riskLevel = document.getElementById('riskLevel');

        // Update stats with animation
        this.animateNumbers(result);

        // Update main display
        const probability = (result.churn_probability * 100).toFixed(1);
        probabilityText.textContent = probability + '%';

        // Set colors based on risk level
        const riskColors = {
            'Low': 'risk-low',
            'Medium': 'risk-medium',
            'High': 'risk-high'
        };

        const colorClass = riskColors[result.risk_level];
        resultHeader.className = `card-header text-white ${colorClass}`;
        probabilityCircle.className = `probability-circle ${colorClass}`;

        // Update text with enhanced messaging
        if (result.will_churn) {
            predictionResult.innerHTML = '⚠️ Customer Likely to Churn';
            riskLevel.innerHTML = `${this.getRiskEmoji(result.risk_level)} ${result.risk_level} risk of customer leaving`;
        } else {
            predictionResult.innerHTML = '✅ Customer Likely to Stay';
            riskLevel.innerHTML = `${this.getRiskEmoji(result.risk_level)} ${result.risk_level} risk of customer leaving`;
        }

        // Show results with enhanced animation
        resultsSection.style.display = 'block';
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            this.addResultsInteractivity(result);
        }, 100);
    }

    animateNumbers(result) {
        const churnProb = document.getElementById('churnProb');
        const confidence = document.getElementById('confidence');
        const riskCategory = document.getElementById('riskCategory');

        // Animate percentage counters
        this.animateCounter(churnProb, 0, result.churn_probability * 100, 1500, '%');
        this.animateCounter(confidence, 0, result.confidence * 100, 1500, '%');
        
        // Animate risk category
        setTimeout(() => {
            riskCategory.textContent = result.risk_level;
            riskCategory.style.animation = 'pulse 1s ease-in-out';
        }, 1000);
    }

    animateCounter(element, start, end, duration, suffix = '') {
        const startTime = performance.now();
        
        const updateCounter = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const current = start + (end - start) * this.easeOutCubic(progress);
            element.textContent = current.toFixed(1) + suffix;
            
            if (progress < 1) {
                requestAnimationFrame(updateCounter);
            }
        };
        
        requestAnimationFrame(updateCounter);
    }

    easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    getRiskEmoji(riskLevel) {
        const emojis = {
            'Low': '🟢',
            'Medium': '🟡',
            'High': '🔴'
        };
        return emojis[riskLevel] || '⚪';
    }

    addResultsInteractivity(result) {
        // Add click-to-copy functionality for results
        const statsItems = document.querySelectorAll('.stat-item');
        statsItems.forEach(item => {
            item.style.cursor = 'pointer';
            item.title = 'Click to copy';
            
            item.addEventListener('click', () => {
                const value = item.querySelector('.stat-value').textContent;
                const label = item.querySelector('.stat-label').textContent;
                navigator.clipboard.writeText(`${label}: ${value}`);
                this.showNotification('Copied to clipboard!', 'info');
            });
        });
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 300px;';
        
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    resetForm() {
        document.getElementById('predictionForm').reset();
        document.getElementById('resultsSection').style.display = 'none';
        
        // Reset progress
        const progressBar = document.querySelector('.progress-indicator');
        if (progressBar) {
            progressBar.style.transform = 'scaleX(0)';
            progressBar.classList.remove('active');
        }
        
        // Reset validation states
        document.querySelectorAll('.form-control, .form-select').forEach(field => {
            field.classList.remove('is-valid', 'is-invalid');
        });
        
        // Reset to default values
        this.setDefaultValues();
        
        this.showNotification('Form reset successfully', 'info');
    }

    setDefaultValues() {
        const defaults = {
            age: '42',
            gender: 'Female',
            geography: 'France',
            creditScore: '619',
            balance: '83807.86',
            estimatedSalary: '101348.88',
            tenure: '2',
            numOfProducts: '1'
        };

        Object.keys(defaults).forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.value = defaults[id];
            }
        });

        document.getElementById('hasCrCard').checked = true;
        document.getElementById('isActiveMember').checked = true;
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.churnPredictor = new ChurnPredictor();
});

// Global reset function for the button
function resetForm() {
    window.churnPredictor.resetForm();
}

// Add some Easter eggs and advanced features
document.addEventListener('keydown', (e) => {
    // Konami code for developer mode
    const konamiCode = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'KeyB', 'KeyA'];
    window.konamiKeys = window.konamiKeys || [];
    window.konamiKeys.push(e.code);
    
    if (window.konamiKeys.length > konamiCode.length) {
        window.konamiKeys.shift();
    }
    
    if (window.konamiKeys.join(',') === konamiCode.join(',')) {
        console.log('🎉 Developer mode activated!');
        window.churnPredictor.showNotification('🎉 Developer mode activated!', 'success');
        // Add some fun developer features here
    }
});