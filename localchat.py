from pywhispercpp.examples.assistant import Assistant
from pywhispercpp.model import Model
from langchain_community.llms import Ollama
from colorama import init, Fore, Style
from gtts import gTTS
import os
import time
import logging

# Initialize colorama
init()

llm = Ollama(model="llama2")
#whisper = Model('base.en', n_threads=6, speed_up=True)
#whisper = Model('base.en', n_threads=6, speed_up=True, print_realtime=False, print_progress=False, print_timestamps=False)

def chatter(inputter):
    print(Fore.CYAN + inputter + Style.RESET_ALL)
    res = llm.predict(inputter)
    print (Fore.RED + Fore.YELLOW + res + Style.RESET_ALL)
    tts = gTTS(res)
    tts.save("output.mp3")
    time.sleep(2)
    #os.system("play -n -c1 synth sin %-12 sin %-9 sin %-5 sin %-2 fade h 0.1 1 0.1")
    os.system("play -q output.mp3")

#my_assistant = Assistant(commands_callback=chatter, n_threads=4, model_log_level=logging.ERROR)
my_assistant = Assistant(model='tiny', silence_threshold=120, commands_callback=chatter, n_threads=4, model_log_level=logging.ERROR)
my_assistant.start()


