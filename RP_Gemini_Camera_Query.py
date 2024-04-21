import vertexai
from vertexai.generative_models import GenerativeModel, Image
from picamera2 import Picamera2
#from time import sleep
import os
from gtts import gTTS
from playsound import playsound
from gpiozero import Button
import asyncio


os.environ["LIBCAMERA_LOG_LEVELS"] = "3"
button = Button(17)
vertexai.init(project='vision-for-the-blind-415812')
model = GenerativeModel('gemini-1.0-pro-vision')

# initialising the camera module 
camera = Picamera2()
camera_config = camera.create_still_configuration(main={"size": (1920, 1080)})
camera.configure(camera_config)
camera.start()

def capture_and_generate_images():
    try:
        camera.capture_file('image.jpg')
        image = Image.load_from_file('image.jpg')
        model_response = model.generate_content(['identify the product being held, be specific with branding if applicable, tell me what the product is and nothing else', image]).text
        return model_response
    except Exception as e:
        print(f'Error: {e}')

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save('output.mp3') # see if i can make this step more efficient using gTTs
    playsound('output.mp3')
    os.remove('output.mp3')

def all_functions():
    model_response = capture_and_generate_images()
    print(model_response)
    text_to_speech(model_response)
    
async def buttonpress():
    #button_pressed = False
    while True:
        button.when_pressed = all_functions
        
print('primed and ready')        
loop = asyncio.get_event_loop()

try:
    loop.run_until_complete(buttonpress())
except KeyboardInterrupt:
    print('interrupted')
finally:
    print('finito')
    camera.stop()