import sounddevice as sd
import numpy as np
import threading

notes = [
    "A0", "A#0", "B0",
    "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1", "A1", "A#1", "B1",
    "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
    "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
    "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6", "A6", "A#6", "B6",
    "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7", "A7", "A#7", "B7",
    "C8"
]

keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
key = keys[0]
octave = 4
major = True

# timesteps & stuff
sample_rate = 44100
T = 10
t = np.linspace(0, T, int(T * sample_rate), False)
stream_active = True

# Fading constants
fade_duration = 0.01 # 10 ms
fade_samples = int(fade_duration * sample_rate)
fade_in = np.linspace(0, 1, fade_samples)
fade_out = np.linspace(1, 0, fade_samples)

current_buffer = None
buffer_postion = 0
buffer_lock = threading.Lock()

def audio_callback(outdata, frames, time, status):
    global current_buffer
    global buffer_postion

    if status:
        print(status)

    with buffer_lock:
        buffer = current_buffer

        if buffer is None:
            outdata.fill(0)
            return

        buffer_length = len(buffer)

        # Fill output
        remaining = frames
        output_position = 0

        while remaining > 0:
            available = buffer_length - buffer_postion
            count = min(remaining, available)

            outdata[output_position:output_position + count, 0] = buffer[buffer_postion:buffer_postion + count]

            buffer_postion += count
            output_position += count
            remaining -= count

            if buffer_postion >= buffer_length:
                buffer_postion = 0

# Prepare and start audio stream
def start_stream():
    t = np.linspace(0, T, int(T * sample_rate), False)
    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=audio_callback)
    stream.start()
    return stream

def set_audio_buffer(buffer):
    global current_buffer
    global buffer_position

    with buffer_lock:
        current_buffer = buffer
        buffer_position = 0

def write_to_stream(stream, buffer):
    stream.write(buffer.astype(np.float32) / 32767)

def set_key(desired_key: str) -> str:
    global key
    global keys

    desired_key.upper()
    key = keys[keys.index(desired_key)]
    slice_index = keys.index(desired_key)
    first_half, second_half = keys[slice_index:], keys[:slice_index]
    keys = first_half + second_half

    return key

def get_chord_note_names(chord_num: int) -> list[str]:
    global keys
    global key

    # Save the original key
    original_key = key

    # Set the key to whatever chord we're building for simplicity
    set_key(keys[0])

    # If we are in a major key we fetch major-specific indices
    if major:
        root = keys[0]
        sharp_first = keys[1]
        second = keys[2]
        sharp_second = keys[3]
        third = keys[4]
        fourth = keys[5]
        sharp_fourth = keys[6]
        fifth = keys[7]
        sharp_fifth = keys[8]
        sixth = keys[9]
        sharp_sixth = keys[10]
        seventh = keys[11]
    else: # Otherwise we're in minor
        root = keys[0]
        sharp_first = keys[1]
        second = keys[2]
        third = keys[3]
        sharp_third = keys[4]
        fourth = keys[5]
        sharp_fourth = keys[6]
        fifth = keys[7]
        sixth = keys[8]
        sharp_sixth = keys[9]
        seventh = keys[10]
        sharp_seventh = keys[11]

    # Restore original key ordering
    set_key(original_key)

    o = str(octave)

    match chord_num:
        case 1:
            return [root + o, third + o, fifth + o]
        case 2:
            return [second + o, fourth + o, sixth + o]
        case 3: 
            return [third + o, fifth + o, seventh + o]
        case 4:
            return [root + o, fourth + o, sixth + o]
        case 5:
            return [second + o, fifth + o, seventh + o]
        case 6:
            return [root + o, third + o, sixth + o]
        case 7:
            return [second + o, fourth + o, seventh + o]
        case _:
            raise ValueError("Value Error: chord_num must be in range 1-7.")
        
def get_frequency_from_note_name(note_name):
    
    n = notes.index(note_name) + 1
    frequency = pow(2, ((n - 49)/12)) * 440
    return frequency

def prepare_chord_buffer(notes: list[str]):

    note_frequencies = [get_frequency_from_note_name(note) for note in notes]

    chord_wave = sum(np.sin(2 * np.pi * frequency * t) for frequency in note_frequencies)

    # Normalize chord to max 1 to prevent clipping
    chord_wave /= np.max(np.abs(chord_wave))

    # Apply fading to minimize clipping on note start
    chord_wave[:fade_samples] *= fade_in
    chord_wave[-fade_samples:] *= fade_out

    return (chord_wave * 0.25).astype(np.float32)

if __name__ == "__main__":

    stream = start_stream()

    print(get_chord_note_names(1))
    print(get_chord_note_names(3))
    print(get_chord_note_names(7))
    print(key)
    print(keys)
    
    chord_numbers = [1, 2, 3, 4, 5, 6, 7]
    chords = []
    
    for number in chord_numbers:
        chords.append(get_chord_note_names(number))
    
    for chord in chords:
        audio_data = prepare_chord_buffer(chord)
        #all_chords.append(audio_data)
        stream.write(audio_data.astype(np.float32) / 32767)