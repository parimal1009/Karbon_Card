// JavaScript for AI Bank Parser Agent Frontend

// Global variables
let currentBankName = '';
let statusCheckInterval = null;
let autoRefreshInterval = null;
let isAutoRefreshEnabled = false;

// API endpoints
const API_BASE = '';
const ENDPOINTS = {
    status: '/api/status',
    upload: '/api/upload',
    generate: '/api/generate',
    download: '/api/download',
    test: '/api/test',
    logs: '/api/logs',
    validate: '/api/validate',
    benchmark: '/api/benchmark',
    parsers: '/api/parsers'
};

// Utility functions
function showLoading(text = 'Processing...') {
    const overlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    loadingText.textContent = text;
    overlay.style.display = 'block';
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = 'none';
}

function showModal(title, content) {
    const modal = document.getElementById('notificationModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalBody = document.getElementById('modalBody');
    
    modalTitle.textContent = title;
    modalBody.innerHTML = content;
    modal.style.display = 'block';
}

function closeModal() {
    const modal = document.getElementById('notificationModal');
    modal.style.display = 'none';
}

function showStatus(element, message, type) {
    element.textContent = message;
    element.className = `status-message ${type}`;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatTime(seconds) {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = (seconds % 60).toFixed(1);
    return `${minutes}m ${remainingSeconds}s`;
}

// File upload functionality
async function uploadFiles() {
    const bankName = document.getElementById('bankName').value.trim();
    const pdfFile = document.getElementById('pdfFile').files[0];
    const csvFile = document.getElementById('csvFile').files[0];
    const statusDiv = document.getElementById('uploadStatus');

    // Validation
    if (!bankName || !pdfFile || !csvFile) {
        showStatus(statusDiv, '❌ Please fill all fields', 'error');
        return;
    }

    if (!pdfFile.name.toLowerCase().endsWith('.pdf')) {
        showStatus(statusDiv, '❌ Please select a valid PDF file', 'error');
        return;
    }

    if (!csvFile.name.toLowerCase().endsWith('.csv')) {
        showStatus(statusDiv, '❌ Please select a valid CSV file', 'error');
        return;
    }

    // Create form data
    const formData = new FormData();
    formData.append('pdf', pdfFile);
    formData.append('csv', csvFile);

    try {
        showLoading('Uploading files...');
        showStatus(statusDiv, '📤 Uploading files...', 'info');

        const response = await fetch(`${API_BASE}${ENDPOINTS.upload}?bank_name=${encodeURIComponent(bankName)}`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            currentBankName = bankName.toLowerCase();
            showStatus(statusDiv, `✅ Files uploaded successfully for ${bankName}!`, 'success');
            
            // Show file details
            const details = `
                <div style="margin-top: 10px; font-size: 0.9rem; color: var(--text-muted);">
                    📄 PDF: ${formatBytes(pdfFile.size)}<br>
                    📊 CSV: ${formatBytes(csvFile.size)}
                </div>
            `;
            statusDiv.innerHTML += details;
            
            // Enable generate button
            document.getElementById('generateBtn').disabled = false;
            
            // Update status bar
            updateStatusBar();
            
        } else {
            showStatus(statusDiv, `❌ Upload failed: ${result.detail}`, 'error');
        }
    } catch (error) {
        showStatus(statusDiv, `❌ Upload error: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// Parser generation functionality
async function generateParser() {
    if (!currentBankName) {
        const bankName = document.getElementById('bankName').value.trim();
        if (!bankName) {
            showModal('Error', '❌ Please upload files first!');
            return;
        }
        currentBankName = bankName.toLowerCase();
    }

    const maxIterations = parseInt(document.getElementById('maxIterations').value);
    const statusDiv = document.getElementById('generateStatus');
    const generateBtn = document.getElementById('generateBtn');
    const progressContainer = document.getElementById('progressContainer');

    try {
        generateBtn.disabled = true;
        progressContainer.style.display = 'block';
        showStatus(statusDiv, '🚀 Starting parser generation...', 'info');

        const response = await fetch(`${API_BASE}${ENDPOINTS.generate}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                bank_name: currentBankName,
                max_iterations: maxIterations
            })
        });

        const result = await response.json();

        if (response.ok) {
            showStatus(statusDiv, `✅ ${result.message}`, 'success');
            startStatusPolling();
        } else {
            showStatus(statusDiv, `❌ ${result.detail}`, 'error');
            generateBtn.disabled = false;
            progressContainer.style.display = 'none';
        }
    } catch (error) {
        showStatus(statusDiv, `❌ Error: ${error.message}`, 'error');
        generateBtn.disabled = false;
        progressContainer.style.display = 'none';
    }
}

// Status polling functionality
function startStatusPolling() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }

    statusCheckInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}${ENDPOINTS.status}`);
            const status = await response.json();

            updateProgress(status);
            updateStatusBar(status);

            if (!status.is_running && (status.status === 'completed' || status.status === 'failed')) {
                clearInterval(statusCheckInterval);
                handleCompletion(status);
            }
        } catch (error) {
            console.error('Status check failed:', error);
        }
    }, 2000); // Check every 2 seconds
}

// Update progress UI
function updateProgress(status) {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const iterationInfo = document.getElementById('iterationInfo');

    progressFill.style.width = `${status.progress}%`;
    
    const statusMessages = {
        'idle': '⏳ Ready to start...',
        'starting': '🔄 Initializing agent...',
        'analyzing': '🔍 Analyzing PDF and CSV...',
        'generating': '⚡ Generating parser code...',
        'testing': '🧪 Testing parser...',
        'correcting': '🔧 Self-correcting code...',
        'optimizing': '⚡ Optimizing performance...',
        'completed': '✅ Parser generated successfully!',
        'failed': '❌ Generation failed'
    };

    progressText.textContent = statusMessages[status.status] || status.status;
    
    if (status.iteration > 0) {
        iterationInfo.textContent = `Iteration ${status.iteration}/${status.max_iterations}`;
    } else {
        iterationInfo.textContent = '';
    }
}

// Update status bar
function updateStatusBar(status = null) {
    const agentStatus = document.getElementById('agentStatus');
    const currentBank = document.getElementById('currentBank');
    const progressPercent = document.getElementById('progressPercent');

    if (status) {
        agentStatus.textContent = status.status.charAt(0).toUpperCase() + status.status.slice(1);
        currentBank.textContent = status.current_bank || 'None';
        progressPercent.textContent = `${status.progress}%`;
    } else if (currentBankName) {
        currentBank.textContent = currentBankName.toUpperCase();
    }
}

// Handle completion
function handleCompletion(status) {
    const statusDiv = document.getElementById('generateStatus');
    const generateBtn = document.getElementById('generateBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const testBtn = document.getElementById('testBtn');
    const benchmarkBtn = document.getElementById('benchmarkBtn');

    generateBtn.disabled = false;

    if (status.status === 'completed') {
        showStatus(statusDiv, '✅ Parser generated successfully!', 'success');
        downloadBtn.disabled = false;
        testBtn.disabled = false;
        benchmarkBtn.disabled = false;
        
        // Show success notification
        showModal('Success! 🎉', `
            <p>Parser for <strong>${currentBankName.toUpperCase()}</strong> has been generated successfully!</p>
            <p>You can now:</p>
            <ul style="margin: 10px 0; padding-left: 20px;">
                <li>💾 Download the parser code</li>
                <li>🧪 Test it against your sample data</li>
                <li>📊 Benchmark its performance</li>
            </ul>
        `);
        
    } else {
        showStatus(statusDiv, `❌ Generation failed: ${status.error}`, 'error');
        document.getElementById('progressContainer').style.display = 'none';
        
        // Show error details
        showModal('Generation Failed ❌', `
            <p>Parser generation for <strong>${currentBankName.toUpperCase()}</strong> failed.</p>
            <p><strong>Error:</strong></p>
            <pre style="background: #f5f5f5; padding: 10px; border-radius: 4px; font-size: 0.9rem; white-space: pre-wrap;">${status.error}</pre>
            <p>Please check the logs for more details and try again.</p>
        `);
    }
}

// Download parser
async function downloadParser() {
    if (!currentBankName) {
        showModal('Error', '❌ No parser to download!');
        return;
    }

    try {
        showLoading('Preparing download...');
        
        const response = await fetch(`${API_BASE}${ENDPOINTS.download}/${currentBankName}`);
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${currentBankName}_parser.py`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showModal('Download Started 📥', `
                <p>Parser code for <strong>${currentBankName.toUpperCase()}</strong> is being downloaded.</p>
                <p>The file <code>${currentBankName}_parser.py</code> should appear in your downloads folder.</p>
            `);
        } else {
            const error = await response.json();
            showModal('Download Failed', `❌ ${error.detail}`);
        }
    } catch (error) {
        showModal('Download Error', `❌ ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Test parser
async function testParser() {
    if (!currentBankName) {
        showModal('Error', '❌ No parser to test!');
        return;
    }

    const resultsDiv = document.getElementById('testResults');
    resultsDiv.innerHTML = '<p style="color: var(--text-muted);">🧪 Testing parser...</p>';

    try {
        showLoading('Testing parser...');
        
        const response = await fetch(`${API_BASE}${ENDPOINTS.test}/${currentBankName}`);
        const result = await response.json();

        if (result.status === 'success') {
            displayTestResults(result, resultsDiv);
        } else {
            resultsDiv.innerHTML = `
                <div class="status-message error">
                    <strong>❌ Test Failed:</strong><br>
                    ${result.error || result.message}
                </div>
            `;
        }
    } catch (error) {
        resultsDiv.innerHTML = `
            <div class="status-message error">
                <strong>❌ Error:</strong> ${error.message}
            </div>
        `;
    } finally {
        hideLoading();
    }
}

// Display test results
function displayTestResults(result, container) {
    const performanceInfo = result.execution_time ? 
        `<small style="color: var(--text-muted);">⏱️ ${formatTime(result.execution_time)} | 💾 ${result.memory_usage?.toFixed(1) || 'N/A'} MB</small>` : '';
    
    const html = `
        <div class="status-message success">
            <strong>✅ Test Passed!</strong><br>
            📊 Rows: ${result.rows} | 📋 Columns: ${result.columns.join(', ')}<br>
            ${performanceInfo}
        </div>
        <h4 style="margin-top: 20px; color: var(--text-primary);">📋 Sample Data (First 5 Rows)</h4>
        <div style="overflow-x: auto; margin-top: 10px;">
            <table>
                <thead>
                    <tr>
                        ${result.columns.map(col => `<th>${col}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${result.sample.map(row => `
                        <tr>
                            ${result.columns.map(col => `<td>${row[col] || ''}</td>`).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
    container.innerHTML = html;
}

// Benchmark parser
async function benchmarkParser() {
    if (!currentBankName) {
        showModal('Error', '❌ No parser to benchmark!');
        return;
    }

    try {
        showLoading('Running benchmark...');
        
        const response = await fetch(`${API_BASE}${ENDPOINTS.benchmark}/${currentBankName}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                bank_name: currentBankName,
                iterations: 3
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showModal('Benchmark Results 📊', `
                <h4>Performance Metrics for ${currentBankName.toUpperCase()}</h4>
                <div style="margin: 15px 0;">
                    <p><strong>⏱️ Average Time:</strong> ${formatTime(result.avg_time)}</p>
                    <p><strong>🚀 Best Time:</strong> ${formatTime(result.min_time)}</p>
                    <p><strong>🐌 Worst Time:</strong> ${formatTime(result.max_time)}</p>
                    <p><strong>💾 Average Memory:</strong> ${result.avg_memory?.toFixed(1) || 'N/A'} MB</p>
                    <p><strong>✅ Success Rate:</strong> ${result.success_rate?.toFixed(1) || 'N/A'}%</p>
                    <p><strong>🔄 Iterations:</strong> ${result.iterations}</p>
                </div>
            `);
        } else {
            showModal('Benchmark Failed', `❌ ${result.error}`);
        }
    } catch (error) {
        showModal('Benchmark Error', `❌ ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Refresh logs
async function refreshLogs() {
    const logsContent = document.getElementById('logsContent');
    logsContent.textContent = 'Loading logs...';

    try {
        const response = await fetch(`${API_BASE}${ENDPOINTS.logs}`);
        const result = await response.json();

        if (result.logs && result.logs.length > 0) {
            logsContent.textContent = result.logs.join('');
        } else {
            logsContent.textContent = 'No logs available.';
        }

        // Auto-scroll to bottom
        const logsContainer = document.getElementById('logsContainer');
        logsContainer.scrollTop = logsContainer.scrollHeight;
    } catch (error) {
        logsContent.textContent = `Error loading logs: ${error.message}`;
    }
}

// Clear logs (visual only)
function clearLogs() {
    const logsContent = document.getElementById('logsContent');
    logsContent.textContent = 'Logs cleared (refresh to reload)';
}

// Toggle auto-refresh for logs
function toggleAutoRefresh() {
    const toggleBtn = document.getElementById('autoRefreshToggle');
    
    if (isAutoRefreshEnabled) {
        clearInterval(autoRefreshInterval);
        isAutoRefreshEnabled = false;
        toggleBtn.textContent = '▶️ Auto';
    } else {
        autoRefreshInterval = setInterval(refreshLogs, 5000);
        isAutoRefreshEnabled = true;
        toggleBtn.textContent = '⏸️ Auto';
    }
}

// Refresh status manually
async function refreshStatus() {
    try {
        const response = await fetch(`${API_BASE}${ENDPOINTS.status}`);
        const status = await response.json();
        updateStatusBar(status);
        
        if (status.is_running) {
            updateProgress(status);
            if (!statusCheckInterval) {
                startStatusPolling();
            }
        }
    } catch (error) {
        console.error('Status refresh failed:', error);
    }
}

// Advanced functionality
function toggleAdvanced() {
    const content = document.getElementById('advancedContent');
    const toggle = document.getElementById('advancedToggle');
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        toggle.textContent = 'Hide';
    } else {
        content.style.display = 'none';
        toggle.textContent = 'Show';
    }
}

async function validateSetup() {
    try {
        showLoading('Validating setup...');
        
        const response = await fetch(`${API_BASE}${ENDPOINTS.validate}`);
        const result = await response.json();
        
        let content = '';
        if (result.valid) {
            content = `
                <div style="color: var(--success-color);">
                    <h4>✅ Setup Validation Passed</h4>
                    <p>All dependencies and configurations are properly set up!</p>
                </div>
            `;
        } else {
            content = `
                <div style="color: var(--error-color);">
                    <h4>❌ Setup Issues Found</h4>
                    <p><strong>Issues:</strong></p>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        ${result.issues.map(issue => `<li>${issue}</li>`).join('')}
                    </ul>
                    ${result.recommendations.length > 0 ? `
                        <p><strong>Recommendations:</strong></p>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            ${result.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    ` : ''}
                </div>
            `;
        }
        
        showModal('System Validation', content);
    } catch (error) {
        showModal('Validation Error', `❌ ${error.message}`);
    } finally {
        hideLoading();
    }
}

function showValidation() {
    validateSetup();
}

function showApiDocs() {
    window.open('/docs', '_blank');
}

function exportLogs() {
    const logsContent = document.getElementById('logsContent').textContent;
    const blob = new Blob([logsContent], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `agent_logs_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

async function listParsers() {
    try {
        showLoading('Loading parsers...');
        
        const response = await fetch(`${API_BASE}${ENDPOINTS.parsers}`);
        const result = await response.json();
        
        if (result.parsers.length === 0) {
            showModal('Parsers List', '<p>No parsers found. Generate some parsers first!</p>');
            return;
        }
        
        const content = `
            <h4>📋 Generated Parsers (${result.total})</h4>
            <div style="margin-top: 15px;">
                ${result.parsers.map(parser => `
                    <div style="border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; margin-bottom: 10px;">
                        <h5 style="margin: 0 0 10px 0; color: var(--primary-color);">${parser.bank_name.toUpperCase()}</h5>
                        <p style="margin: 5px 0; font-size: 0.9rem; color: var(--text-muted);">
                            📄 Size: ${formatBytes(parser.file_size)} | 
                            📅 Created: ${new Date(parser.created_at).toLocaleDateString()} |
                            ${parser.test_status ? `✅ ${parser.test_status}` : '❓ Not tested'}
                        </p>
                    </div>
                `).join('')}
            </div>
        `;
        
        showModal('Generated Parsers', content);
    } catch (error) {
        showModal('Error', `❌ Failed to load parsers: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function showAbout() {
    showModal('About AI Bank Parser Agent', `
        <div style="text-align: center;">
            <h3>🤖 AI Bank Parser Agent</h3>
            <p style="margin: 15px 0;">An intelligent AI agent that automatically generates bank statement parsers using advanced language models.</p>
            
            <h4 style="margin-top: 20px;">🛠️ Technologies</h4>
            <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin: 10px 0;">
                <span class="badge">FastAPI</span>
                <span class="badge">LangChain</span>
                <span class="badge">Groq</span>
                <span class="badge">HuggingFace</span>
            </div>
            
            <h4 style="margin-top: 20px;">✨ Features</h4>
            <ul style="text-align: left; margin: 10px 0; padding-left: 20px;">
                <li>🔄 Self-correcting AI agent</li>
                <li>📊 Real-time progress tracking</li>
                <li>🧪 Automated testing</li>
                <li>📈 Performance benchmarking</li>
                <li>🌐 Modern web interface</li>
            </ul>
            
            <p style="margin-top: 20px; color: var(--text-muted);">
                Built with ❤️ for <strong>Karbon Card AI Challenge</strong>
            </p>
        </div>
    `);
}

// Event listeners and initialization
document.addEventListener('DOMContentLoaded', function() {
    // Initial setup
    refreshLogs();
    refreshStatus();
    
    // Close modal when clicking outside
    window.onclick = function(event) {
        const modal = document.getElementById('notificationModal');
        if (event.target === modal) {
            closeModal();
        }
    };
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeModal();
            hideLoading();
        }
    });
    
    // Auto-refresh logs every 30 seconds
    setInterval(() => {
        if (!isAutoRefreshEnabled) {
            refreshLogs();
        }
    }, 30000);
    
    console.log('🤖 AI Bank Parser Agent initialized');
});
