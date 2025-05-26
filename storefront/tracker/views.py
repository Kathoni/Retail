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
from .models import AIInsight, Business, BusinessMetrics, ProfitPrediction, Transaction
from django.contrib.auth import login
from .forms import BusinessForm, RegisterForm
@login_required
def dashboard(request):
    # try:
    #     business = Business.objects.get(owner=request.user)
    # except Business.DoesNotExist:
    #     return redirect('das')  # Redirect to business creation if none

    # today = timezone.now().date()

    # # Get today's metrics
    # try:
    #     today_metrics = BusinessMetrics.objects.get(business=business, date=today)
    # except BusinessMetrics.DoesNotExist:
    #     today_metrics = calculate_daily_metrics(business, today)

    # # Get recent transactions
    # recent_transactions = Transaction.objects.filter( 
    #     business=business,
    #     timestamp__date=today
    # )[:10]

    # # Get active AI insights
    # ai_insights = AIInsight.objects.filter(
    #     business=business,
    #     is_active=True
    # )[:5]

    # # Get profit prediction for next 7 days
    # predictions = ProfitPrediction.objects.filter(
    #     business=business,
    #     prediction_date__gte=today,
    #     prediction_date__lte=today + timedelta(days=7)
    # )

    # context = {
    #     'business': business,
    #     'today_metrics': today_metrics,
    #     'recent_transactions': recent_transactions,
    #     'ai_insights': ai_insights,
    #     'predictions': predictions,
    # }

    return render(request, 'dashboard.html')
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

@csrf_exempt
def add_transaction_api(request):
    """API endpoint for adding transactions"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            business = get_object_or_404(Business, owner=request.user)
            
            transaction = Transaction.objects.create(
                business=business,
                transaction_type=data['type'],
                amount=data['amount'],
                description=data['description'],
                input_method=data.get('input_method', 'manual'),
                quantity=data.get('quantity'),
                supplier=data.get('supplier', ''),
                customer_type=data.get('customer_type', ''),
                location=data.get('location', ''),
            )
            
            # Trigger AI analysis
            generate_ai_insights(business)
            
            return JsonResponse({
                'success': True,
                'transaction_id': transaction.id,
                'ai_insight': get_transaction_insight(transaction)
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

def get_transaction_insight(transaction):
    """Generate immediate AI insight for a transaction"""
    insights = {
        'sale': [
            '✅ Good timing (peak hour)' if 12 <= transaction.hour_of_day <= 18 else '📊 Normal timing',
            '🎯 High margin item' if transaction.amount > 50 else '📊 Average sale value',
            '🔥 Popular item today' if transaction.day_of_week < 5 else '📊 Weekend sale'
        ],
        'expense': [
            '⚠️ Could be optimized' if transaction.amount > 30 else '📊 Normal expense',
            '🚨 Above average cost' if transaction.amount > 100 else '💡 Consider bulk buying',
            '📊 Regular business expense'
        ],
        'loss': [
            '🚨 Pattern detected - review inventory',
            '⚠️ Wastage increasing',
            '💡 Consider smaller orders'
        ]
    }
    
    return insights.get(transaction.transaction_type, ['📊 Transaction recorded'])[0]

def generate_ai_insights(business):
    """Generate AI insights based on transaction patterns"""
    # Get recent transaction data
    recent_transactions = Transaction.objects.filter(
        business=business,
        timestamp__gte=timezone.now() - timedelta(days=30)
    )
    
    # Detect wastage patterns
    detect_wastage_patterns(business, recent_transactions)
    
    # Detect timing opportunities
    detect_timing_opportunities(business, recent_transactions)
    
    # Detect cost optimization opportunities
    detect_cost_optimization(business, recent_transactions)

def detect_wastage_patterns(business, transactions):
    """Detect wastage patterns and create insights"""
    wastage_by_day = {}
    
    for transaction in transactions.filter(transaction_type='loss'):
        day = transaction.day_of_week
        wastage_by_day[day] = wastage_by_day.get(day, 0) + float(transaction.amount)
    
    if wastage_by_day:
        worst_day = max(wastage_by_day, key=wastage_by_day.get)
        worst_amount = wastage_by_day[worst_day]
        
        if worst_amount > 20:  # Threshold for significant wastage
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            AIInsight.objects.get_or_create(
                business=business,
                insight_type='wastage',
                title=f'High Wastage on {days[worst_day]}s',
                defaults={
                    'severity': 'high' if worst_amount > 50 else 'medium',
                    'description': f'Your wastage on {days[worst_day]}s averages ${worst_amount:.2f}. Consider reducing inventory by 15-20% on this day.',
                    'potential_savings': worst_amount * 0.7,
                    'confidence_score': 0.8,
                }
            )

def detect_timing_opportunities(business, transactions):
    """Detect peak sales times and missed opportunities"""
    sales_by_hour = {}
    
    for transaction in transactions.filter(transaction_type='sale'):
        hour = transaction.hour_of_day
        sales_by_hour[hour] = sales_by_hour.get(hour, 0) + float(transaction.amount)
    
    if sales_by_hour:
        peak_hour = max(sales_by_hour, key=sales_by_hour.get)
        peak_amount = sales_by_hour[peak_hour]
        avg_amount = sum(sales_by_hour.values()) / len(sales_by_hour)
        
        if peak_amount > avg_amount * 1.5:  # 50% above average
            AIInsight.objects.get_or_create(
                business=business,
                insight_type='timing',
                title=f'Peak Sales at {peak_hour}:00 Hour',
                defaults={
                    'severity': 'medium',
                    'description': f'Sales spike {((peak_amount/avg_amount-1)*100):.0f}% at {peak_hour}:00. Ensure adequate inventory during this time.',
                    'potential_revenue': (peak_amount - avg_amount) * 0.3,
                    'confidence_score': 0.7,
                }
            )

def detect_cost_optimization(business, transactions):
    """Detect cost optimization opportunities"""
    transport_costs = transactions.filter(
        transaction_type='expense',
        description__icontains='transport'
    ).aggregate(total=Sum('amount'))['total'] or 0
    
    if transport_costs > 100:  # Monthly threshold
        AIInsight.objects.get_or_create(
            business=business,
            insight_type='cost',
            title='Transport Cost Optimization',
            defaults={
                'severity': 'medium',
                'description': f'Monthly transport costs: ${transport_costs}. Consider consolidating trips or finding closer suppliers.',
                'potential_savings': transport_costs * 0.2,
                'confidence_score': 0.6,
            }
        )

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
            login(request, user)  # log in user immediately after registration
            return redirect('dashboard')  # redirect to your dashboard or homepage
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})

@login_required
def create_business(request):
    if hasattr(request.user, 'business'):
        return redirect('dashboard')  # User already has a business

    if request.method == 'POST':
        form = BusinessForm(request.POST)
        if form.is_valid():
            business = form.save(commit=False)
            business.owner = request.user
            business.save()
            return redirect('dashboard')
    else:
        form = BusinessForm()
    return render(request, 'create_business.html', {'form': form})