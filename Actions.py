import subprocess
from enum import Enum
from functools import partial

class KEYS(Enum):
    LEFT = "Left"
    RIGHT = "Right"
    UP = "Up"
    DOWN = "Down"

def send_key_press(key):
    subprocess.call(["xdotool", "key", key])

ACTIONS = {
    0:    partial(send_key_press, KEYS.LEFT.value),
    1:   partial(send_key_press, KEYS.RIGHT.value),
    2:      partial(send_key_press, KEYS.UP.value),
    3:    partial(send_key_press, KEYS.DOWN.value),
}