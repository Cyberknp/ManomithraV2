import requests
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.shortcuts import redirect, render

try:
    import cv2
    import numpy as np
    import tensorflow as tf

    CV2_AVAILABLE = True
    TF_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    TF_AVAILABLE = False
    cv2 = None
    np = None
    tf = None


json_path = "C:/projects/Manomitra/Django/opencv/emotiondetector.json"
weights_path = "C:/projects/Manomitra/Django/opencv/emotiondetector.h5"


def load_model():
    if not TF_AVAILABLE:
        print("TensorFlow not available. Emotion detection disabled.")
        return None
    try:
        print("Loading model...")
        with open(json_path, "r") as json_file:
            model_json = json_file.read()
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(weights_path)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return None


model = load_model()

if CV2_AVAILABLE:
    haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_file)
else:
    haar_file = None
    face_cascade = None


labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}


# Function to process emotion detection
def detect_emotion():
    if not CV2_AVAILABLE or not TF_AVAILABLE or model is None:
        print(
            "Emotion detection unavailable (missing dependencies or model). Defaulting to neutral."
        )
        return "neutral"

    webcam = cv2.VideoCapture(0)
    ret, frame = webcam.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        webcam.release()
        return None

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        try:
            face = gray[y : y + h, x : x + w]
            face_resized = cv2.resize(face, (48, 48))
            img = np.array(face_resized).reshape(1, 48, 48, 1) / 255.0  # Normalize

            # Predict the emotion
            pred = model(img, training=False)  # Direct TensorFlow model prediction
            prediction_label = labels[np.argmax(pred.numpy())]

            # Print the detected emotion
            print(f"Detected Emotion: {prediction_label}")
            webcam.release()
            return prediction_label
        except Exception as e:
            print(f"Error processing face: {e}")
            continue

    webcam.release()
    return None


# Function to run the chatbot with detected emotion
def run_chatbot(user_input, emotion):
    system_instruction = (
        "You are a chatbot pretending to be a human. Respond with human-like emotions "
        "but focus on delivering a message based on the user's input. Use the person's "
        "expression as an additional cue but do not emphasize your own expressions. "
        "If the user's message conveys one emotion, but their face shows another, "
        "acknowledge the difference. For example, if the message is happy but their "
        "face appears sad, say something like, 'You seem happy, but your face suggests "
        "you might be feeling a bit down.' Avoid using phrases like '**' or making "
        "your own expressions the main focus."
    )

    prompt = f"{system_instruction}\n\nUser: {user_input}\nEmotion detected: {emotion}\nChatbot:"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma:2b",
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Please make sure Ollama is running."
    except requests.exceptions.Timeout:
        return "Error: The model took too long to respond. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"


# Home page - no login required
def home1(request):
    return render(request, "mainPage.html")


# Sign-up page - no login required
def signup(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        password = request.POST.get("password")
        username = request.POST.get("username")

        if User.objects.filter(username=username).exists():
            messages.error(
                request, "Username already exists. Please choose a different one."
            )
        elif User.objects.filter(email=email).exists():
            messages.error(
                request, "Email already registered. Please use a different email."
            )
        else:
            user = User.objects.create_user(
                username=username, password=password, email=email, first_name=name
            )
            user.save()
            messages.success(request, "Sign-Up successful! You can now log in.")
            return redirect("login_view")

    return render(request, "signup.html")  # Corrected path to 'signup.html'


# Login page - no login required
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        # Authenticate user
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("/chatbot")
        else:
            messages.error(request, "Invalid username or password. Please try again.")
            return render(
                request, "login.html"
            )  # Render the login page again with an error message

    return render(request, "login.html")


def aboutus(request):
    return render(request, "aboutus.html")


# Chatbot page - login required
@login_required
def chatbot(request):
    if request.method == "POST":
        user_input = request.POST.get("user_input")

        # Run the emotion detection in the background
        emotion = detect_emotion()  # Get the detected emotion from the camera
        if emotion is None:
            emotion = "neutral"  # Default emotion if none is detected

        # Run the chatbot with emotion and user input
        response = run_chatbot(user_input, emotion)

        return JsonResponse({"response": response, "success": True})

    return render(request, "chatbot.html")


# Logout page - no login required
def logout_view(request):
    logout(request)
    return redirect("/login")
