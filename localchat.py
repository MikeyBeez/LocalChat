from pywhispercpp.examples.assistant import Assistant
from pywhispercpp.model import Model
from langchain_community.llms import Ollama
from colorama import init, Fore, Style
import os
import time
import logging

# Initialize colorama
init()

llm = Ollama(model="llama2")

def chatter(inputter):
    print(Fore.CYAN + inputter + Style.RESET_ALL)
    res = llm.invoke(inputter)
    print (Fore.RED + Fore.YELLOW + res + Style.RESET_ALL)

my_assistant = Assistant(model='tiny', silence_threshold=40, commands_callback=chatter, n_threads=4, model_log_level=logging.ERROR)
my_assistant.start()


