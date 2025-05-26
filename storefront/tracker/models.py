# models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import datetime, timedelta
import json

class Business(models.Model):
    owner = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    business_type = models.CharField(max_length=100, choices=[
        ('retail', 'Retail Store'),
        ('food', 'Food Vendor'),
        ('market', 'Market Trader'),
        ('service', 'Service Provider'),
    ])
    location = models.CharField(max_length=200)
    phone = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} - {self.owner.username}"

class Category(models.Model):
    name = models.CharField(max_length=100)
    type = models.CharField(max_length=20, choices=[
        ('income', 'Income'),
        ('expense', 'Expense'),
    ])
    color = models.CharField(max_length=7, default="#3498db")  # Hex color
    
    class Meta:
        verbose_name_plural = "Categories"
    
    def __str__(self):
        return f"{self.name} ({self.type})"

class Transaction(models.Model):
    TRANSACTION_TYPES = [
        ('sale', 'Sale'),
        ('expense', 'Expense'),
        ('loss', 'Loss/Wastage'),
    ]
    
    INPUT_METHODS = [
        ('manual', 'Manual Entry'),
        ('voice', 'Voice Input'),
        ('photo', 'Photo Scan'),
        ('api', 'API Integration'),
    ]
    
    business = models.ForeignKey(Business, on_delete=models.CASCADE, related_name='transactions', default=1)
    transaction_type = models.CharField(max_length=20, choices=TRANSACTION_TYPES)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField()
    input_method = models.CharField(max_length=20, choices=INPUT_METHODS, default='manual')
    
    # Metadata for AI analysis
    quantity = models.FloatField(null=True, blank=True)
    unit_price = models.DecimalField(max_digits=8, decimal_places=2, null=True, blank=True)
    supplier = models.CharField(max_length=200, blank=True)
    customer_type = models.CharField(max_length=50, blank=True)  # regular, new, bulk, etc.
    
    # Location and timing data
    location = models.CharField(max_length=200, blank=True)
    weather = models.CharField(max_length=50, blank=True)
    day_of_week = models.IntegerField(default=0)  # 0=Monday, 6=Sunday
    hour_of_day = models.IntegerField(default=12)
    
    timestamp = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.business.name} - {self.transaction_type} - ${self.amount}"
    
    def save(self, *args, **kwargs):
        # Set timestamp if not already set
        if not self.timestamp:
            self.timestamp = timezone.now()
        
        # Auto-populate day and hour
        self.day_of_week = self.timestamp.weekday()
        self.hour_of_day = self.timestamp.hour
        super().save(*args, **kwargs)

class AIInsight(models.Model):
    SEVERITY_LEVELS = [
        ('low', 'Low Priority'),
        ('medium', 'Medium Priority'),
        ('high', 'High Priority'),
        ('critical', 'Critical'),
    ]
    
    INSIGHT_TYPES = [
        ('wastage', 'Wastage Pattern'),
        ('timing', 'Timing Optimization'),
        ('pricing', 'Pricing Strategy'),
        ('inventory', 'Inventory Management'),
        ('cost', 'Cost Optimization'),
        ('revenue', 'Revenue Opportunity'),
    ]
    
    business = models.ForeignKey(Business, on_delete=models.CASCADE, related_name='ai_insights')
    insight_type = models.CharField(max_length=20, choices=INSIGHT_TYPES)
    severity = models.CharField(max_length=10, choices=SEVERITY_LEVELS)
    title = models.CharField(max_length=200)
    description = models.TextField()
    
    # Financial impact
    potential_savings = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    potential_revenue = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    # Related data for context
    related_transactions = models.ManyToManyField(Transaction, blank=True)
    confidence_score = models.FloatField(default=0.0)  # 0-1 confidence in this insight
    
    is_active = models.BooleanField(default=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at', '-severity']
    
    def __str__(self):
        return f"{self.business.name} - {self.title} ({self.severity})"

class ProfitPrediction(models.Model):
    business = models.ForeignKey(Business, on_delete=models.CASCADE, related_name='predictions')
    prediction_date = models.DateField()
    predicted_revenue = models.DecimalField(max_digits=10, decimal_places=2)
    predicted_expenses = models.DecimalField(max_digits=10, decimal_places=2)
    predicted_profit = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Confidence intervals
    revenue_low = models.DecimalField(max_digits=10, decimal_places=2)
    revenue_high = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Factors influencing prediction
    weather_factor = models.FloatField(default=1.0)
    seasonal_factor = models.FloatField(default=1.0)
    event_factor = models.FloatField(default=1.0)
    trend_factor = models.FloatField(default=1.0)
    
    model_accuracy = models.FloatField(default=0.0)  # Historical accuracy of this prediction
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['business', 'prediction_date']
        ordering = ['prediction_date']
    
    def __str__(self):
        return f"{self.business.name} - {self.prediction_date} - ${self.predicted_profit}"

class BusinessMetrics(models.Model):
    """Daily aggregated metrics for quick dashboard loading"""
    business = models.ForeignKey(Business, on_delete=models.CASCADE, related_name='metrics')
    date = models.DateField()
    
    # Daily totals
    total_revenue = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_expenses = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_profit = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_transactions = models.IntegerField(default=0)
    
    # Performance metrics
    avg_transaction_value = models.DecimalField(max_digits=8, decimal_places=2, default=0)
    profit_margin = models.FloatField(default=0.0)
    
    # Efficiency metrics
    wastage_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    wastage_percentage = models.FloatField(default=0.0)
    
    # AI Risk Score (0-10)
    risk_score = models.FloatField(default=5.0)
    
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['business', 'date']
        ordering = ['-date']
    
    def __str__(self):
        return f"{self.business.name} - {self.date} - ${self.total_profit}"

class VoiceInput(models.Model):
    """Store voice input data for processing and improvement"""
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    audio_file = models.FileField(upload_to='voice_inputs/', null=True, blank=True)
    transcribed_text = models.TextField()
    confidence = models.FloatField(default=0.0)
    processed = models.BooleanField(default=False)
    
    # Extracted transaction data
    extracted_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    extracted_type = models.CharField(max_length=20, blank=True)
    extracted_description = models.TextField(blank=True)
    
    created_transaction = models.ForeignKey(Transaction, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.business.name} - Voice Input - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

class PhotoReceipt(models.Model):
    """Store photo receipt data for OCR processing"""
    business = models.ForeignKey(Business, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='receipts/')
    processed = models.BooleanField(default=False)
    
    # OCR extracted data
    raw_text = models.TextField(blank=True)
    extracted_data = models.JSONField(default=dict)  # Store structured data
    
    # Confidence scores for different extractions
    amount_confidence = models.FloatField(default=0.0)
    vendor_confidence = models.FloatField(default=0.0)
    date_confidence = models.FloatField(default=0.0)
    
    created_transactions = models.ManyToManyField(Transaction, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.business.name} - Receipt - {self.created_at.strftime('%Y-%m-%d')}"

class MLModel(models.Model):
    """Track ML model versions and performance"""
    MODEL_TYPES = [
        ('profit_prediction', 'Profit Prediction'),
        ('loss_detection', 'Loss Pattern Detection'),
        ('demand_forecast', 'Demand Forecasting'),
        ('price_optimization', 'Price Optimization'),
    ]
    
    model_type = models.CharField(max_length=30, choices=MODEL_TYPES)
    version = models.CharField(max_length=20)
    accuracy = models.FloatField(default=0.0)
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    
    # Model file storage
    model_file = models.FileField(upload_to='ml_models/', null=True, blank=True)
    training_data_count = models.IntegerField(default=0)
    
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['model_type', 'version']
    
    def __str__(self):
        return f"{self.model_type} v{self.version} - {self.accuracy:.2%}"