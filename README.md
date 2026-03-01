# ManoMitra Version 2

A private and empathetic AI companion for your mental well-being.It hears ,it feels ,it understands and its with you along the entire journey of your healing. 

In a world that feels more connected yet often more isolating, we wanted to create something that truly listens. ManoMitra was born from that idea: an AI companion designed to understand you on a deeper level, offering a safe space for your thoughts and feelings.

## About the Project

ManoMitra is not just another chatbot. It's an attempt to create a digital friend that can sense how you're feeling without you having to say a word. By optionally using your camera and microphone, it can pick up on visual cues and the tone of your voice, allowing for a conversation that is much more in tune with your actual emotional state.

Crucially, this project is built with privacy at its core. It uses Ollama to run powerful language models directly on your own machine. This means your conversations, your expressions, and your data remain completely private and under your control.

## Key Features

* **Understands You Beyond Words:** Talk, type, or turn on your camera. ManoMitra is designed to understand your tone of voice and facial expressions, catching the nuances that text alone can miss.
* **Truly Private Conversations:** Because the AI runs locally on your machine, nothing you say or show is sent to the cloud. Your thoughts are yours alone.
* **A Friend That Remembers:** ManoMitra keeps track of your past conversations to provide a continuous and personal experience. You won't have to repeat yourself, just like with a real friend.
* **Intelligent and Meaningful Chat:** Powered by modern large language models, the conversations are thoughtful, context-aware, and genuinely helpful.

## Technologies We Used

We built ManoMitra using a blend of modern and powerful technologies to bring this vision to life.

* **Language Model Engine:** Ollama, for running local models like Llama 3 or Mistral.
* **Backend:** Django, for building a robust and scalable server application.
* **Computer Vision:** OpenCV and MediaPipe for real-time facial expression analysis.
* **Audio Processing:** Librosa and the SpeechRecognition library to analyze vocal tone and transcribe speech.
* **Frontend:** React.js, for a modern and responsive user interface that can access the camera and microphone.

## Getting Started

Interested in running your own version of ManoMitra? Here’s how to get it set up.

### Prerequisites

You'll need to have these tools installed on your computer before you begin:
* Python 3.8+
* Node.js and npm
* Git
* Ollama (with a model like `llama3` already downloaded)

### Installation Guide

1.  First, clone a copy of the project to your local machine:
    ```sh
    git clone [https://github.com/Pra-navK/ManoMitra.git](https://github.com/Pra-navK/ManoMitra.git)
    cd ManoMitra
    ```

2.  Next, set up the backend server (usually in a `/backend` or root project folder):
    ```sh
    # Navigate to your Django project directory
    pip install -r requirements.txt
    ```

3.  Then, set up the frontend application (usually in a `/frontend` folder):
    ```sh
    # Navigate to your React project directory
    npm install
    ```

4.  Finally, configure your local environment variables. Create a `.env` file in the main Django project folder and add your settings.
    ```
    SECRET_KEY='YOUR_DJANGO_SECRET_KEY'
    DEBUG=True
    DATABASE_URL='YOUR_DATABASE_URL'
    ```

## How to Run the Application

Once everything is installed, you can start the application with these two commands.

1.  Start the backend Django server from its project directory:
    ```sh
    python manage.py runserver
    ```
    The API should now be running at `http://127.0.0.1:8000`.

2.  In a separate terminal, start the frontend React application from its directory:
    ```sh
    npm start
    ```
    You can now open [http://localhost:3000](http://localhost:3000) in your browser to start chatting.

## Our Vision for the Future

ManoMitra is just getting started. Here are some of the ideas we're excited to explore next:
* Creating dedicated mobile apps for iOS and Android.
* Exploring integration with wearable devices to better understand stress indicators.
* Offering an option to connect with certified human counselors.
* Adding support for multiple languages.

## How to Contribute

This is a project driven by a passion for mental wellness and technology. If this idea resonates with you, we would love your help. Contributions of any kind are welcome, whether it's improving the code, fixing a bug, or suggesting a new feature.

Please feel free to open an issue or submit a pull request.


## Contact

Cyberknp [K Nagendra Pai ] - [GitHub Profile](https://github.com/cyberknp)

Project Link: [https://github.com/Cyberknp/Manomithra.git]
