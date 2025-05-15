from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from .models import model, GraphFramemrk
import torch
from pathlib import Path
from PreProcess import HandSignSingle

def base(request):
    return render(request, "index.html")

def tosign(request):
    return render(request, "tosign.html")

def totext(request):
    return render(request, "totext.html")

@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(video.name, video)
        file_url = fs.url(filename)
        return JsonResponse({'url': file_url})
    return JsonResponse({'error': 'No video uploaded'}, status=400)

def prediction(request):
    model.eval()
    video_path = request.GET.get('video')
    filename = Path(video_path)
    out_csv = f'data/{filename}.csv'
    GraphFramemrk(video_path=video_path, out_csv=out_csv)
    input_ids, edge, batch = HandSignSingle(csv_path=out_csv)
    with torch.no_grad():
        output = model(input_ids, edge, batch)
        prediction = torch.argmax(output, dim=1)
        

