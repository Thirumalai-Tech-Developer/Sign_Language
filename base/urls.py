from django.urls import path
from . import views

urlpatterns = [
    path("/", views.base, name="index"),
    path('tosign/', views.tosign, name='tosign'),
    path('totext/', views.totext, name='totext'),
    path('upload_video/', views.upload_video, name='upload_video'),
]