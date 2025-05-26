"""
URL configuration for storefront project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'tracker'  # <<<<< ADD THIS


urlpatterns = [
    path('admin/', admin.site.urls),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('create_business/', views.create_business, name='create_business'),
    path('accounts/login/', auth_views.LoginView.as_view(), name='login'),
    path('login/', auth_views.LoginView.as_view()),
    path('register/', views.register, name='register'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('add_transaction_api/', views.add_transaction_api, name='add_transaction_api'),
    path('get_dashboard_data/', views.get_dashboard_data, name='get_dashboard_data'),
    path('process-voice/', views.process_voice, name='process_voice'),
    path('process-voice-text/', views.process_voice_text, name='process_voice_text'),
    path('update_transaction/', views.update_transaction, name='update_transaction'),
    path('delete_transaction/<int:transaction_id>/', views.delete_transaction, name='delete_transaction'),
]
