#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pathlib
import threading
from queue import Queue
import numpy as np
import pydub
from langchain.callbacks.base import BaseCallbackHandler
from pywhispercpp.examples.assistant import Assistant
from langchain.llms import Ollama
from colorama import init, Fore, Style
from gtts import gTTS
import os
from langchain.callbacks.manager import CallbackManager
import sounddevice as sd
from threading import Thread

# colorama
init()

####### GLOBAL VARS #######################
OLLAMA_MODEL = "llama2:7b-chat-q4_0"
WHISPER_MODEL = 'base.en'
WHISPER_N_THREADS = 4
SILENCE_THRESHOLD = 36 # more time to speak
BREAK_SENTENCE_TOKENS = ['!', '.', '?']  # break the sentence when a token from this list is found
TTS_FOLDER = './tts_files/' # where to store the tts files
OUTPUT_TTS_FILE_EXTENSION = 'mp3'  # wav
tts_queue = Queue()
playback_event = threading.Event()
###########################################

class LLMCallbackHandler(BaseCallbackHandler):

    def __init__(self):
        super().__init__()
        self.i = 0  # sentence number
        self.sentence = ""


    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(Fore.RED + Fore.YELLOW + token + Style.RESET_ALL, end='', flush=True)
        self.sentence += token
        if token in BREAK_SENTENCE_TOKENS:
            # run a thread to generate the tts sentence while the llm still generating tokens
            thread = Thread(target=generate_tts_sentence, args=(self.sentence, self.i))
            thread.start()
            self.i += 1
            self.sentence = ''

    def on_llm_start(self, serialized, prompts, **kwargs,):
        # create tts folder
        os.makedirs(TTS_FOLDER, exist_ok=True)
        # clear tts folder
        files = os.listdir(TTS_FOLDER)
        for file in files:
            os.remove(os.path.join(TTS_FOLDER, file))
        self.i = 0
        self.sentence = ''



    def on_llm_end(self, response, **kwargs):
        print()

def generate_tts_sentence(sentence: str, i):
    tts = gTTS(sentence)
    file = (pathlib.Path(TTS_FOLDER) / f"{i}.{OUTPUT_TTS_FILE_EXTENSION}").resolve()
    tts.save(str(file))
    tts_queue.put(str(file))
    # Signal the playback event
    if tts_queue and not playback_event.is_set():
        playback_event.set()

def play_manager():
    while True:
        # Wait for the playback event to be set
        playback_event.wait()
        file = tts_queue.get()
        thread = threading.Thread(target=play, args=(file,))
        thread.start()
        thread.join()
        if tts_queue.qsize() == 0:
            # clear the event if nothing left in the queue
            playback_event.clear()

def play(file):
    def _file_to_numpy(f, normalized=False):
        a = pydub.AudioSegment.from_file(f)
        y = np.array(a.get_array_of_samples())
        if a.channels == 2:
            y = y.reshape((-1, 2))
        if normalized:
            return a.frame_rate, np.float32(y) / 2 ** 15
        else:
            return a.frame_rate, y

    fs, y = _file_to_numpy(file)
    sd.play(y, fs)
    sd.wait()
    os.remove(file)

#
# template_messages = [
#     SystemMessage(content="You are a helpful assistant, keep your answers concise and short"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     HumanMessagePromptTemplate.from_template("{text}"),
# ]
# prompt_template = ChatPromptTemplate.from_messages(template_messages)
#
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

llm = Ollama(
    model=OLLAMA_MODEL,
    callback_manager=CallbackManager([LLMCallbackHandler()]),
)

def chatter(text):
    if not playback_event.is_set():
        print(Fore.CYAN + text + Style.RESET_ALL)
        res = llm(text)

def main():
    play_manager_thread = threading.Thread(target=play_manager)
    play_manager_thread.start()
    my_assistant = Assistant(model=WHISPER_MODEL,
                             n_threads=WHISPER_N_THREADS,
                             commands_callback=chatter,
                             silence_threshold=SILENCE_THRESHOLD,
                             # model_log_level=logging.ERROR, # no logs
                             )

    my_assistant.start()


if __name__ == '__main__':
    main()
