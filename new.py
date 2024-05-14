import speech_recognition as sr
import pyttsx3
import os


engine=pyttsx3.init()   #making an instance(object) of pyttsx3 class so that we can use its associated methods like say,etc


def say(text):
    engine.say(text)
    engine.runAndWait()


def listen():
    r=sr.Recognizer()  #making an instance of speechrecognition class to listen what I say 
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        r.pause_threshold=1
        audio=r.listen(source)
        query=r.recognize_google_cloud(audio,language="en_US")
if __name__== '__main__':
    print("Hello World")
    say("Hello I am Jarvis A I ..") 
    while True:
        print("listening....")
        text=listen()
        say(text)