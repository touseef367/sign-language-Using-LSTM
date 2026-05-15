from gtts import gTTS
import os

text = input("Enter the text you want to convert to sound: ")

# Create a gTTS object with the text and desired language
tts = gTTS(text=text, lang='en', slow=False)

# Save the speech as an MP3 file
output_file = 'output.mp3'
tts.save(output_file)

# Play the generated sound file immediately after saving it
os.system('start ' + output_file)  # For Windows
# For macOS, use the following line instead:
# os.system('afplay ' + output_file)
# For Linux, you can use 'aplay':
# os.system('aplay ' + output_file)
