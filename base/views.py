from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import VideoFile
from django.core.files.storage import FileSystemStorage

def base(request):
    return render(request, "index.html")

def tosign(request):
    return render(request, "tosign.html")

def totext(request):
    return render(request, "totext.html")

def upload_video(request):
    if request.method == 'POST' and request.FILES['video_file']:
        video_file = request.FILES['video_file']
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        file_url = fs.url(filename)