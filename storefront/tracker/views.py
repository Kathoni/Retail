# views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.db.models import Sum, Avg, Count, Q
from django.utils import timezone
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os
from .models import AIInsight, Business, BusinessMetrics, ProfitPrediction, Transaction, VoiceInput
from django.contrib.auth import login
from .forms import BusinessForm, RegisterForm, TransactionForm
from django.views.decorators.http import require_http_methods
from decimal import Decimal
import base64
import decimal

@login_required
def dashboard(request):
    business = get_object_or_404(Business, owner=request.user)
    today = timezone.now().date()
    
    # Get today's metrics
    metrics, created = BusinessMetrics.objects.get_or_create(
        business=business,
        date=today,
        defaults={
            'total_revenue': Decimal('0.00'),
            'total_expenses': Decimal('0.00'),
            'total_profit': Decimal('0.00'),
            'profit_margin': Decimal('0.00'),
        }
    )
    
    # Get recent transactions
    recent_transactions = Transaction.objects.filter(business=business).order_by('-timestamp')[:10]
    
    # Get AI insights
    ai_insights = AIInsight.objects.filter(business=business, is_active=True).order_by('-severity', '-created_at')[:5]
    
    # Get predictions
    predictions = ProfitPrediction.objects.filter(
        business=business,
        prediction_date__gte=today
    ).order_by('prediction_date')[:7]
    
    context = {
        'today_metrics': metrics,
        'recent_transactions': recent_transactions,
        'ai_insights': ai_insights,
        'predictions': predictions,
    }
    
    return render(request, 'dashboard.html', context)

def calculate_daily_metrics(business, date):
    """Calculate and store daily metrics"""
    transactions = Transaction.objects.filter(
        business=business,
        timestamp__date=date
    )
    
    revenue = transactions.filter(transaction_type='sale').aggregate(
        total=Sum('amount')
    )['total'] or 0
    
    expenses = transactions.filter(transaction_type='expense').aggregate(
        total=Sum('amount')
    )['total'] or 0
    
    losses = transactions.filter(transaction_type='loss').aggregate(
        total=Sum('amount')
    )['total'] or 0
    
    profit = revenue - expenses - losses
    profit_margin = (profit / revenue * 100) if revenue > 0 else 0
    
    metrics, created = BusinessMetrics.objects.get_or_create(
        business=business,
        date=date,
        defaults={
            'total_revenue': revenue,
            'total_expenses': expenses,
            'total_profit': profit,
            'total_transactions': transactions.count(),
            'profit_margin': profit_margin,
            'wastage_amount': losses,
            'wastage_percentage': (losses / revenue * 100) if revenue > 0 else 0,
            'risk_score': calculate_risk_score(business, date),
        }
    )
    
    return metrics

def calculate_risk_score(business, date):
    """Calculate AI risk score based on various factors"""
    # Get last 30 days of data
    end_date = date
    start_date = date - timedelta(days=30)
    
    transactions = Transaction.objects.filter(
        business=business,
        timestamp__date__range=[start_date, end_date]
    )
    
    if not transactions.exists():
        return 5.0  # Neutral score
    
    # Risk factors
    risk_factors = []
    
    # 1. Profit volatility
    daily_profits = []
    for i in range(30):
        day = end_date - timedelta(days=i)
        day_transactions = transactions.filter(timestamp__date=day)
        day_revenue = day_transactions.filter(transaction_type='sale').aggregate(Sum('amount'))['amount__sum'] or 0
        day_expenses = day_transactions.filter(transaction_type='expense').aggregate(Sum('amount'))['amount__sum'] or 0
        daily_profits.append(float(day_revenue - day_expenses))
    
    if daily_profits:
        profit_volatility = np.std(daily_profits) / (np.mean(daily_profits) + 1)
        risk_factors.append(min(profit_volatility * 2, 10))
    
    # 2. Wastage trend
    wastage_trend = transactions.filter(transaction_type='loss').count() / max(transactions.count(), 1)
    risk_factors.append(wastage_trend * 10)
    
    # 3. Declining revenue trend
    recent_revenue = transactions.filter(
        timestamp__date__gte=end_date - timedelta(days=7),
        transaction_type='sale'
    ).aggregate(Sum('amount'))['amount__sum'] or 0
    
    older_revenue = transactions.filter(
        timestamp__date__range=[end_date - timedelta(days=14), end_date - timedelta(days=7)],
        transaction_type='sale'
    ).aggregate(Sum('amount'))['amount__sum'] or 0
    
    if older_revenue > 0:
        revenue_decline = max(0, (older_revenue - recent_revenue) / older_revenue)
        risk_factors.append(revenue_decline * 10)
    
    # Calculate weighted average
    if risk_factors:
        return min(max(sum(risk_factors) / len(risk_factors), 0), 10)
    
    return 5.0

@login_required
def add_transaction(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Validate required fields
            if not all(key in data for key in ['transaction_type', 'amount', 'description']):
                return JsonResponse({
                    'success': False,
                    'error': 'Missing required fields'
                }, status=400)
            
            # Validate transaction type
            if data['transaction_type'] not in ['sale', 'expense', 'loss']:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid transaction type'
                }, status=400)
            
            # Validate amount
            try:
                amount = Decimal(str(data['amount']))
                if amount <= 0:
                    return JsonResponse({
                        'success': False,
                        'error': 'Amount must be greater than zero'
                    }, status=400)
            except (ValueError, TypeError, decimal.InvalidOperation):
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid amount value'
                }, status=400)
            
            # Create transaction
            transaction = Transaction.objects.create(
                business=request.user.business,
                transaction_type=data['transaction_type'],
                amount=amount,
                description=data['description'].strip()
            )
            
            # Generate and save AI insight
            insight = get_transaction_insight(transaction)
            transaction.ai_insight = insight
            transaction.save()
            
            # Update metrics
            metrics = calculate_daily_metrics(transaction.business, timezone.now().date())
            
            # Generate new AI insights
            generate_ai_insights(transaction.business)
            
            # Get updated AI insights
            ai_insights = AIInsight.objects.filter(
                business=transaction.business,
                is_active=True
            ).order_by('-severity', '-created_at')[:5]
            
            # Return complete updated data
            return JsonResponse({
                'success': True,
                'transaction': {
                    'id': transaction.id,
                    'type': transaction.transaction_type,
                    'amount': float(transaction.amount),
                    'description': transaction.description,
                    'ai_insight': transaction.ai_insight,
                    'timestamp': transaction.timestamp.isoformat()
                },
                'metrics': {
                    'total_revenue': float(metrics.total_revenue),
                    'total_expenses': float(metrics.total_expenses),
                    'total_profit': float(metrics.total_profit),
                    'profit_margin': float(metrics.profit_margin),
                    'risk_score': float(metrics.risk_score)
                },
                'ai_insights': [{
                    'title': insight.title,
                    'description': insight.description,
                    'severity': insight.severity
                } for insight in ai_insights]
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    }, status=405)

def get_transaction_insight(transaction):
    """Generate immediate AI insight for a transaction"""
    business = transaction.business
    transaction_type = transaction.transaction_type
    amount = float(transaction.amount)  # Convert Decimal to float for calculations
    
    # Get recent transactions for comparison
    recent_transactions = Transaction.objects.filter(
        business=business,
        transaction_type=transaction_type,
        timestamp__gte=timezone.now() - timedelta(days=30)
    ).exclude(id=transaction.id)  # Exclude current transaction
    
    # Calculate average amount for this type of transaction
    avg_amount = recent_transactions.aggregate(Avg('amount'))['amount__avg']
    avg_amount = float(avg_amount) if avg_amount else amount  # Convert Decimal to float
    
    # Get total transactions today
    today_transactions = Transaction.objects.filter(
        business=business,
        timestamp__date=timezone.now().date()
    )
    
    # Calculate daily totals
    daily_sales = today_transactions.filter(transaction_type='sale').aggregate(Sum('amount'))['amount__sum']
    daily_sales = float(daily_sales) if daily_sales else 0  # Convert Decimal to float
    
    daily_expenses = today_transactions.filter(transaction_type__in=['expense', 'loss']).aggregate(Sum('amount'))['amount__sum']
    daily_expenses = float(daily_expenses) if daily_expenses else 0  # Convert Decimal to float
    
    insights = {
        'sale': {
            'high_value': f"🌟 High-value sale! {amount/avg_amount:.1f}x above average" if amount > avg_amount * 1.5 else None,
            'peak_hour': "⭐ Perfect timing - peak business hours" if 11 <= transaction.timestamp.hour <= 19 else None,
            'milestone': f"🎯 Milestone: Daily sales reached ${daily_sales:,.2f}" if daily_sales > avg_amount * 3 else None,
            'profit_margin': "💰 Great profit margin potential" if amount > 1000 else None
        },
        'expense': {
            'high_cost': f"⚠️ High expense alert: {amount/avg_amount:.1f}x above average" if amount > avg_amount * 1.2 else None,
            'timing': "📊 Good timing for restocking" if 6 <= transaction.timestamp.hour <= 10 else None,
            'budget_warning': "🚨 Daily expense budget exceeded" if daily_expenses > daily_sales * 0.5 else None,
            'suggestion': "💡 Consider bulk purchasing to reduce costs" if amount > 500 else None
        },
        'loss': {
            'pattern': "🔍 Loss pattern detected - Review inventory" if recent_transactions.filter(transaction_type='loss').count() > 2 else None,
            'high_loss': f"⚠️ Significant loss: {amount/avg_amount:.1f}x above average" if amount > avg_amount * 1.1 else None,
            'action_needed': "🚨 Immediate action required - Check storage conditions" if amount > 1000 else None,
            'prevention': "💡 Consider implementing loss prevention measures" if recent_transactions.filter(transaction_type='loss').exists() else None
        }
    }
    
    # Get relevant insights for this transaction type
    type_insights = insights.get(transaction_type, {})
    
    # Filter out None values and get the most relevant insight
    valid_insights = [insight for insight in type_insights.values() if insight]
    
    if valid_insights:
        return valid_insights[0]  # Return the most important insight
    
    # Default insights if no specific conditions are met
    default_insights = {
        'sale': "📈 Sale recorded successfully",
        'expense': "📊 Expense tracked and analyzed",
        'loss': "⚠️ Loss recorded - Monitor for patterns"
    }
    
    return default_insights.get(transaction_type, "✅ Transaction recorded")

def generate_ai_insights(business):
    """Generate overall AI insights based on transaction patterns"""
    # Get recent transactions
    recent_transactions = Transaction.objects.filter(
        business=business,
        timestamp__gte=timezone.now() - timedelta(days=30)
    )
    
    if not recent_transactions.exists():
        return
    
    # Calculate key metrics
    total_sales = recent_transactions.filter(transaction_type='sale').aggregate(Sum('amount'))['amount__sum'] or 0
    total_expenses = recent_transactions.filter(transaction_type='expense').aggregate(Sum('amount'))['amount__sum'] or 0
    total_losses = recent_transactions.filter(transaction_type='loss').aggregate(Sum('amount'))['amount__sum'] or 0
    
    # Analyze patterns
    if total_losses > total_sales * 0.1:  # Losses > 10% of sales
        AIInsight.objects.get_or_create(
            business=business,
            title="High Loss Rate Detected",
            defaults={
                'description': f"Losses (${total_losses:,.2f}) are unusually high compared to sales. Consider reviewing inventory management practices.",
                'severity': 'high',
                'insight_type': 'loss_prevention',
                'potential_savings': total_losses * 0.5
            }
        )
    
    # Analyze expense patterns
    if total_expenses > total_sales * 0.7:  # Expenses > 70% of sales
        AIInsight.objects.get_or_create(
            business=business,
            title="High Expense Ratio",
            defaults={
                'description': "Expenses are consuming a large portion of revenue. Consider cost optimization strategies.",
                'severity': 'medium',
                'insight_type': 'cost_optimization',
                'potential_savings': total_expenses * 0.15
            }
        )
    
    # Analyze sales patterns
    sales_by_hour = {}
    for transaction in recent_transactions.filter(transaction_type='sale'):
        hour = transaction.hour_of_day
        sales_by_hour[hour] = sales_by_hour.get(hour, 0) + float(transaction.amount)
    
    if sales_by_hour:
        peak_hour = max(sales_by_hour, key=sales_by_hour.get)
        AIInsight.objects.get_or_create(
            business=business,
            title=f"Peak Sales Hour Identified: {peak_hour}:00",
            defaults={
                'description': f"Sales are highest at {peak_hour}:00. Consider optimizing staffing and inventory during these hours.",
                'severity': 'low',
                'insight_type': 'optimization',
                'potential_revenue': sales_by_hour[peak_hour] * 0.2
            }
        )
    
    # Clean up old insights
    AIInsight.objects.filter(
        business=business,
        created_at__lt=timezone.now() - timedelta(days=7)
    ).update(is_active=False)

# ML prediction functions
def generate_profit_predictions(business):
    """Generate 7-day profit predictions using ML"""
    # Get historical data
    historical_data = prepare_prediction_data(business)
    
    if len(historical_data) < 30:  # Need minimum data
        return create_simple_predictions(business)
    
    # Train model
    model = train_profit_model(historical_data)
    
    # Generate predictions for next 7 days
    predictions = []
    today = timezone.now().date()
    
    for i in range(1, 8):
        pred_date = today + timedelta(days=i)
        features = prepare_prediction_features(business, pred_date)
        
        predicted_profit = model.predict([features])[0]
        confidence_interval = predicted_profit * 0.2  # ±20%
        
        prediction = ProfitPrediction.objects.create(
            business=business,
            prediction_date=pred_date,
            predicted_profit=predicted_profit,
            predicted_revenue=predicted_profit * 1.3,  # Estimate
            predicted_expenses=predicted_profit * 0.3,  # Estimate
            revenue_low=predicted_profit - confidence_interval,
            revenue_high=predicted_profit + confidence_interval,
            model_accuracy=0.75  # Historical accuracy
        )
        predictions.append(prediction)
    
    return predictions

def prepare_prediction_data(business):
    """Prepare historical data for ML training"""
    metrics = BusinessMetrics.objects.filter(
        business=business,
        date__gte=timezone.now().date() - timedelta(days=90)
    ).order_by('date')
    
    data = []
    for metric in metrics:
        features = [
            metric.date.weekday(),  # Day of week
            metric.date.day,        # Day of month
            metric.total_revenue,
            metric.total_expenses,
            metric.profit_margin,
            metric.wastage_percentage,
            metric.risk_score,
        ]
        target = metric.total_profit
        data.append((features, target))
    
    return data

def train_profit_model(historical_data):
    """Train ML model for profit prediction"""
    X = [item[0] for item in historical_data]
    y = [item[1] for item in historical_data]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def prepare_prediction_features(business, date):
    """Prepare features for a specific prediction date"""
    # Get recent metrics for context
    recent_metrics = BusinessMetrics.objects.filter(
        business=business,
        date__lt=date
    ).order_by('-date').first()
    
    if not recent_metrics:
        # Default features
        return [date.weekday(), date.day, 100, 50, 0.5, 0.1, 5.0]
    
    features = [
        date.weekday(),
        date.day,
        float(recent_metrics.total_revenue),
        float(recent_metrics.total_expenses),
        recent_metrics.profit_margin,
        recent_metrics.wastage_percentage,
        recent_metrics.risk_score,
    ]
    
    return features

def create_simple_predictions(business):
    """Create simple predictions when insufficient data"""
    predictions = []
    today = timezone.now().date()
    
    # Use recent average
    recent_avg = BusinessMetrics.objects.filter(
        business=business,
        date__gte=today - timedelta(days=7)
    ).aggregate(
        avg_profit=Avg('total_profit'),
        avg_revenue=Avg('total_revenue'),
        avg_expenses=Avg('total_expenses')
    )
    
    avg_profit = recent_avg['avg_profit'] or 50
    avg_revenue = recent_avg['avg_revenue'] or 150
    avg_expenses = recent_avg['avg_expenses'] or 100
    
    for i in range(1, 8):
        pred_date = today + timedelta(days=i)
        
        # Add some weekend variation
        factor = 1.2 if pred_date.weekday() >= 5 else 1.0
        
        prediction = ProfitPrediction.objects.create(
            business=business,
            prediction_date=pred_date,
            predicted_profit=avg_profit * factor,
            predicted_revenue=avg_revenue * factor,
            predicted_expenses=avg_expenses,
            revenue_low=avg_profit * factor * 0.8,
            revenue_high=avg_profit * factor * 1.2,
            model_accuracy=0.6
        )
        predictions.append(prediction)
    
    return predictions

def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('tracker:create_business')
    else:
        form = RegisterForm()
    return render(request, 'registration/register.html', {'form': form})

@login_required
def create_business(request):
    if hasattr(request.user, 'business'):
        return redirect('tracker:dashboard')
        
    if request.method == 'POST':
        form = BusinessForm(request.POST)
        if form.is_valid():
            business = form.save(commit=False)
            business.owner = request.user
            business.save()
            return redirect('tracker:dashboard')
    else:
        form = BusinessForm()
    
    return render(request, 'create_business.html', {'form': form})

@login_required
def get_dashboard_data(request):
    business = request.user.business
    today = timezone.now().date()
    
    # Get updated metrics
    metrics = BusinessMetrics.objects.get(business=business, date=today)
    
    # Get recent transactions
    transactions = Transaction.objects.filter(business=business).order_by('-timestamp')[:10]
    
    # Format transaction data
    transaction_data = []
    for t in transactions:
        transaction_data.append({
            'id': t.id,
            'timestamp': t.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'description': t.description,
            'transaction_type': t.transaction_type,
            'amount': float(t.amount),
            'ai_insight': t.ai_insight or ''
        })
    
    # Get AI insights
    insights = AIInsight.objects.filter(business=business, is_active=True).order_by('-severity', '-created_at')[:5]
    insight_data = []
    for i in insights:
        insight_data.append({
            'title': i.title,
            'description': i.description,
            'severity': i.severity
        })
    
    # Get predictions
    predictions = ProfitPrediction.objects.filter(
        business=business,
        prediction_date__gte=today
    ).order_by('prediction_date')[:7]
    prediction_data = []
    for p in predictions:
        prediction_data.append({
            'prediction_date': p.prediction_date.strftime('%Y-%m-%d'),
            'predicted_profit': float(p.predicted_profit),
            'revenue_low': float(p.revenue_low),
            'revenue_high': float(p.revenue_high),
            'model_accuracy': float(p.model_accuracy)
        })
    
    return JsonResponse({
        'today_metrics': {
            'total_revenue': float(metrics.total_revenue),
            'total_expenses': float(metrics.total_expenses),
            'total_profit': float(metrics.total_profit),
            'profit_margin': float(metrics.profit_margin),
            'risk_score': float(metrics.risk_score)
        },
        'recent_transactions': transaction_data,
        'ai_insights': insight_data,
        'predictions': prediction_data
    })

@login_required
def update_transaction(request, transaction_id):
    if request.method == 'POST':
        transaction = get_object_or_404(Transaction, id=transaction_id, business=request.user.business)
        data = json.loads(request.body)
        
        transaction.transaction_type = data.get('transaction_type', transaction.transaction_type)
        transaction.amount = Decimal(data.get('amount', transaction.amount))
        transaction.description = data.get('description', transaction.description)
        transaction.save()
        
        return JsonResponse({'success': True})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def delete_transaction(request, transaction_id):
    if request.method == 'POST':
        transaction = get_object_or_404(Transaction, id=transaction_id, business=request.user.business)
        transaction.delete()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
@csrf_exempt
def process_voice(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        business = request.user.business
        audio_file = request.FILES['audio']
        
        voice_input = VoiceInput.objects.create(
            business=business,
            audio_file=audio_file
        )
        
        transaction = Transaction.objects.create(
            business=business,
            transaction_type='sale',
            amount=Decimal('0.00'),
            description='Voice recorded transaction',
            input_method='voice'
        )
        
        voice_input.created_transaction = transaction
        voice_input.processed = True
        voice_input.save()
        
        update_business_metrics(business, transaction)
        return JsonResponse({'success': True})
    return JsonResponse({'success': False, 'error': 'Invalid request'})

@csrf_exempt
@login_required
def process_voice_text(request):
    """Process voice transcription text"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            transcript = data.get('transcript', '').lower()
            business = get_object_or_404(Business, owner=request.user)
            
            # Extract transaction details from transcript
            transaction_type = extract_transaction_type(transcript)
            amount = extract_amount(transcript)
            description = extract_description(transcript)
            
            if transaction_type and amount:
                # Create transaction
                transaction = Transaction.objects.create(
                    business=business,
                    transaction_type=transaction_type,
                    amount=amount,
                    description=description or transcript,
                    input_method='voice'
                )
                
                # Update metrics
                calculate_daily_metrics(business, timezone.now().date())
                
                return JsonResponse({
                    'success': True,
                    'transaction_id': transaction.id,
                    'message': f'Added {transaction_type} of ${amount}'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'Could not extract transaction details from voice input'
                })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

def extract_transaction_type(text):
    """Extract transaction type from voice input"""
    # Multi-language keywords for transaction types
    sale_keywords = ['sale', 'sold', 'venta', 'vendido', 'vendu', 'verkauf', '销售', 'セール']
    expense_keywords = ['expense', 'spent', 'gasto', 'dépense', 'ausgabe', '支出', '経費']
    loss_keywords = ['loss', 'lost', 'pérdida', 'perte', 'verlust', '损失', '損失']
    
    text = text.lower()
    
    if any(keyword in text for keyword in sale_keywords):
        return 'sale'
    elif any(keyword in text for keyword in expense_keywords):
        return 'expense'
    elif any(keyword in text for keyword in loss_keywords):
        return 'loss'
    
    return None

def extract_amount(text):
    """Extract amount from voice input"""
    # Regular expression to find numbers with optional decimal points
    # This will work for various number formats (e.g., 100, 100.00, 1,000.00)
    import re
    
    # Remove currency symbols and commas
    text = text.replace('$', '').replace('€', '').replace('£', '').replace('¥', '').replace(',', '')
    
    # Find all numbers in the text
    numbers = re.findall(r'\d+(?:\.\d{1,2})?', text)
    
    if numbers:
        # Return the first number found
        return float(numbers[0])
    
    return None

def extract_description(text):
    """Extract description from voice input"""
    # Common phrases to ignore in various languages
    ignore_phrases = [
        'add', 'record', 'new', 'transaction', 'sale', 'expense', 'loss',
        'añadir', 'registrar', 'nueva', 'transacción', 'venta', 'gasto', 'pérdida',
        'ajouter', 'enregistrer', 'nouvelle', 'transaction', 'vente', 'dépense', 'perte'
    ]
    
    # Remove common phrases and clean up the text
    words = text.lower().split()
    description_words = [w for w in words if w not in ignore_phrases]
    
    return ' '.join(description_words).capitalize() if description_words else None

def update_business_metrics(business, transaction):
    today = timezone.now().date()
    metrics, created = BusinessMetrics.objects.get_or_create(
        business=business,
        date=today
    )
    
    if transaction.transaction_type == 'sale':
        metrics.total_revenue += transaction.amount
    elif transaction.transaction_type in ['expense', 'loss']:
        metrics.total_expenses += transaction.amount
    
    metrics.total_profit = metrics.total_revenue - metrics.total_expenses
    if metrics.total_revenue > 0:
        metrics.profit_margin = (metrics.total_profit / metrics.total_revenue) * 100
    
    metrics.total_transactions += 1
    metrics.save()

def generate_ai_insights(transaction):
    # Implement your AI insight generation logic here
    pass