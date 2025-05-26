// CSRF Token handling for Django
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Add a new transaction
        function addTransaction(type) {
            const amount = prompt(`Enter amount for ${type}:`);
    if (!amount || isNaN(parseFloat(amount))) {
        showNotification('Error: Please enter a valid amount', 'error');
        return;
    }

            const description = prompt('Enter description:');
    if (!description) {
        showNotification('Error: Please enter a description', 'error');
        return;
    }
            
    const data = {
                    type: type,
                    amount: parseFloat(amount),
        description: description
    };

    // Show loading notification
    showNotification('Adding transaction...', 'info');

    fetch('/tracker/add_transaction_api/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            showNotification(`${type} of $${amount} added successfully!`, 'success');
            updateDashboardData();
        } else {
            showNotification('Error: ' + (data.error || 'Failed to add transaction'), 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error: Failed to add transaction. Please try again.', 'error');
    });
}

// Update dashboard data without page reload
function updateDashboardData() {
    fetch('/tracker/get_dashboard_data/')
        .then(response => response.json())
        .then(data => {
            // Update metrics
            updateMetric('revenue', data.today_metrics.total_revenue, data.today_metrics.revenue_trend);
            updateMetric('expenses', data.today_metrics.total_expenses, data.today_metrics.expense_trend);
            updateMetric('profit', data.today_metrics.total_profit, data.today_metrics.profit_margin);
            updateMetric('risk', data.today_metrics.risk_score);

            // Update transactions table
            updateTransactionsTable(data.recent_transactions);

            // Update insights
            updateInsights(data.ai_insights);

            // Update predictions
            updatePredictions(data.predictions);
        })
        .catch(error => {
            console.error('Error updating dashboard:', error);
        });
}

// Update individual metric
function updateMetric(type, value, trend = null) {
    const formatter = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    });

    switch(type) {
        case 'revenue':
            document.querySelector('.profit-card .metric-value').innerHTML = 
                `${formatter.format(value || 0)}${getTrendIndicator(trend)}`;
            break;
        case 'expenses':
            document.querySelector('.loss-card .metric-value').innerHTML = 
                `${formatter.format(value || 0)}${getTrendIndicator(trend)}`;
            break;
        case 'profit':
            const profitElement = document.querySelector('.card:nth-child(3) .metric-value');
            profitElement.innerHTML = formatter.format(value || 0);
            profitElement.style.color = (value >= 0) ? '#4CAF50' : '#f44336';
            break;
        case 'risk':
            const riskElement = document.querySelector('.card:nth-child(4) .metric-value');
            riskElement.innerHTML = value ? `${value.toFixed(1)}/10` : '-/10';
            riskElement.style.color = getRiskColor(value);
            break;
    }
}

// Get trend indicator HTML
function getTrendIndicator(trend) {
    if (!trend) return '';
    return trend > 0 
        ? '<span class="trend-indicator trend-up">↗</span>' 
        : '<span class="trend-indicator trend-down">↘</span>';
}

// Get risk color based on score
function getRiskColor(score) {
    if (!score) return '#666';
    if (score < 4) return '#4CAF50';
    if (score < 7) return '#FFA726';
    return '#f44336';
}

// Update transactions table
function updateTransactionsTable(transactions) {
    const tbody = document.querySelector('.transactions-table tbody');
    if (!transactions || transactions.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; padding: 20px;">
                    No transactions recorded yet. Use the Quick Actions above to add your first transaction.
                </td>
            </tr>`;
        return;
    }

    tbody.innerHTML = transactions.map(t => `
        <tr>
            <td>${new Date(t.timestamp).toLocaleTimeString()}</td>
            <td>${t.description}</td>
            <td>${t.transaction_type.charAt(0).toUpperCase() + t.transaction_type.slice(1)}</td>
            <td class="amount-${t.transaction_type === 'sale' ? 'positive' : 'negative'}">
                ${t.transaction_type === 'sale' ? '+' : ''}$${t.amount.toFixed(2)}
            </td>
            <td>${t.ai_insight || ''}</td>
        </tr>
    `).join('');
}

// Show notification
function showNotification(message, type = 'success') {
            const notification = document.createElement('div');
            notification.innerHTML = message;
    
    const colors = {
        success: 'linear-gradient(135deg, #4CAF50, #45a049)',
        error: 'linear-gradient(135deg, #f44336, #d32f2f)',
        info: 'linear-gradient(135deg, #2196F3, #1976D2)'
    };

            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
        background: ${colors[type]};
                color: white;
                padding: 15px 25px;
                border-radius: 10px;
                font-weight: 600;
                z-index: 1000;
                animation: slideIn 0.3s ease;
            `;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 3000);
        }

// Photo input placeholder
function startPhotoInput() {
    showNotification('📸 Photo capture feature coming soon!', 'info');
}

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
    // Set up periodic updates
    setInterval(updateDashboardData, 30000); // Update every 30 seconds
        });

        // Add CSS animation for notifications
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);