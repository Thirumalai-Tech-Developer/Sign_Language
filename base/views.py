from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from .models import GTv1GraphConfig, GTv1GraphModel, GraphFramemrk
import torch
import base.tokenizer as tokenizer
from .PreProcess import HandSignSingle
import os
import json
from gtts import gTTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEDIA_FOLDER = settings.MEDIA_FOLDER
TOKEN_PATH = settings.TOKENIZER_PATH
MODEL_PATH = settings.MODEL_PATH

config = GTv1GraphConfig()
model = GTv1GraphModel(config)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)

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

@csrf_exempt
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
        dataset = HandSignSingle(csv_path=out_csv)
        input_ids, edge, batch = dataset[0]

        # Model prediction
        with torch.no_grad():
            output = model(input_ids, edge, batch)
            if output is not None and output.numel() > 0:
                prediction_id = torch.argmax(output, dim=1).item()
            else:
                prediction_id = -1  # or some default/failure indicator


        prediction_id = str(prediction_id)
        predicted_label = tokenizer.tokenize(prediction_id, token_path=TOKEN_PATH)

        tts = gTTS(predicted_label)
        audio_filename = f"{latest_video.stem}.mp3"
        audio_path = os.path.join(f"{MEDIA_FOLDER}/voice", audio_filename)
        print(audio_path)
        tts.save(audio_path)

        return JsonResponse({
            'prediction': predicted_label,
            'video': latest_video.name,
            'audio_url': settings.MEDIA_URL + f"voice/{audio_filename}",
        })

    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def play_video(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '').strip()
            text = text.lower()
            print("Received text:", text)

            video_filename = text + ".mp4"
            video_dir = os.path.join(settings.MEDIA_ROOT, "text_video")
            
            for video in os.listdir(video_dir):
                if video == video_filename:
                    video_url = os.path.join(settings.MEDIA_URL, "text_video", video)
                    return JsonResponse({'video': video_url})

            return JsonResponse({'error': 'Video not found'}, status=404)   

        except Exception as e:
            print("Error:", str(e))
            return JsonResponse({'error': 'Invalid request'}, status=400)

    return JsonResponse({'error': 'Invalid method'}, status=405)
