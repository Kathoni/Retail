 // Simulated Django-like functionality
        let transactions = [];
        let isListening = false;

        function addTransaction(type) {
            const amount = prompt(`Enter amount for ${type}:`);
            const description = prompt('Enter description:');
            
            if (amount && description) {
                const transaction = {
                    type: type,
                    amount: parseFloat(amount),
                    description: description,
                    timestamp: new Date().toLocaleTimeString(),
                    aiInsight: generateAIInsight(type, parseFloat(amount), description)
                };
                
                transactions.unshift(transaction);
                updateDashboard();
                showNotification(`${type} of $${amount} added successfully!`);
            }
        }

        function generateAIInsight(type, amount, description) {
            const insights = {
                sale: [
                    '✅ Good timing (peak hour)',
                    '🎯 High margin item',
                    '📊 Average sale value',
                    '🔥 Popular item today'
                ],
                expense: [
                    '⚠️ Could be optimized',
                    '📊 Normal expense',
                    '🚨 Above average cost',
                    '💡 Consider bulk buying'
                ]
            };
            
            return insights[type][Math.floor(Math.random() * insights[type].length)];
        }

        function startVoiceInput() {
            if (isListening) return;
            
            isListening = true;
            const btn = document.querySelector('.voice-btn');
            btn.innerHTML = '🎤 Listening<span class="loading-dots"></span>';
            btn.classList.add('pulse');
            
            // Simulate voice recognition
            setTimeout(() => {
                const voiceCommands = [
                    { type: 'sale', amount: 35, description: 'Sold mangoes to customer' },
                    { type: 'expense', amount: 12, description: 'Bought plastic bags' },
                    { type: 'sale', amount: 58, description: 'Bulk sale to restaurant' },
                    { type: 'expense', amount: 25, description: 'Fuel for transport' }
                ];
                
                const randomCommand = voiceCommands[Math.floor(Math.random() * voiceCommands.length)];
                
                const confirmed = confirm(`I heard: "${randomCommand.description}" for $${randomCommand.amount}. Add this ${randomCommand.type}?`);
                
                if (confirmed) {
                    transactions.unshift({
                        ...randomCommand,
                        timestamp: new Date().toLocaleTimeString(),
                        aiInsight: generateAIInsight(randomCommand.type, randomCommand.amount, randomCommand.description)
                    });
                    updateDashboard();
                    showNotification(`Voice ${randomCommand.type} added successfully!`);
                }
                
                btn.innerHTML = '🎤 Voice Input';
                btn.classList.remove('pulse');
                isListening = false;
            }, 3000);
        }

        function startPhotoInput() {
            showNotification('📸 Photo capture opening... (Feature simulated)');
            setTimeout(() => {
                const photoResults = [
                    { type: 'expense', amount: 45.50, description: 'Wholesale vegetables receipt' },
                    { type: 'sale', amount: 28.75, description: 'Cash register receipt' },
                    { type: 'expense', amount: 15.25, description: 'Supplier payment receipt' }
                ];
                
                const result = photoResults[Math.floor(Math.random() * photoResults.length)];
                
                const confirmed = confirm(`Photo processed: "${result.description}" for $${result.amount}. Add this ${result.type}?`);
                
                if (confirmed) {
                    transactions.unshift({
                        ...result,
                        timestamp: new Date().toLocaleTimeString(),
                        aiInsight: '📱 Added via photo scan'
                    });
                    updateDashboard();
                    showNotification(`Photo ${result.type} added successfully!`);
                }
            }, 2000);
        }

        function updateDashboard() {
            // This would connect to Django backend in real implementation
            console.log('Dashboard updated with new transaction data');
        }

        function showNotification(message) {
            const notification = document.createElement('div');
            notification.innerHTML = message;
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: linear-gradient(135deg, #4CAF50, #45a049);
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

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Smart Trader Analytics Dashboard Loaded');
            
            // Simulate real-time updates
            setInterval(() => {
                // This would fetch new data from Django backend
                console.log('Checking for updates...');
            }, 30000);
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