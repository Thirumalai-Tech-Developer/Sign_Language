from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from .models import model, GraphFramemrk
import torch
import base.tokenizer as tokenizer
from .PreProcess import HandSignSingle

MEDIA_FOLDER = settings.MEDIA_FOLDER
TOKEN_PATH = settings.TOKENIZER_PATH

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

csrf_exempt
def prediction(request):
    if request.method == 'POST':
        model.eval()

        # Get the latest uploaded video file
        video_files = list(MEDIA_FOLDER.glob('*.mp4'))  # Adjust if you're using other formats
        if not video_files:
            return JsonResponse({'error': 'No video found in media folder'}, status=404)

        latest_video = max(video_files, key=os.path.getctime)
        print(f"Latest uploaded video: {latest_video.name}")

        # Convert video to landmark CSV
        out_csv = f'data/{latest_video.stem}.csv'
        GraphFramemrk(video_path=str(latest_video), out_csv=out_csv)

        # Prepare graph data from CSV
        input_ids, edge, batch = HandSignSingle(csv_path=out_csv)

        # Model prediction
        with torch.no_grad():
            output = model(input_ids, edge, batch)
            prediction_id = torch.argmax(output, dim=1).item()

        prediction_id = str(prediction_id)
        predicted_label = tokenizer(prediction_id)

        return JsonResponse({
            'prediction': predicted_label,
            'video': latest_video.name
        })

    return JsonResponse({'error': 'Invalid request method'}, status=405)