import speech_recognition as sr

WAVS = ['1.wav', '2.wav', '3.wav', '4.wav', '5.wav', '6.wav', '7.wav']


def rec_audio(wav):
    print('----')
    print(wav)
    r = sr.Recognizer()
    audio = None
    with sr.AudioFile(wav) as source:
        audio = r.record(source)
    # sphinx:
    try:
        print('sphinx:', str(r.recognize_sphinx(audio)).lower())
    except Exception as e:
        print(u'error: {e}'.format(e=e))
    # google:
    try:
        print('google:', str(r.recognize_google(audio)).lower())
    except Exception as e:
        print(u'error: {e}'.format(e=e))


for file in WAVS:
    rec_audio(file)
