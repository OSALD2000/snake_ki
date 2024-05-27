from Xlib import X, display
from PIL import Image
import time
from enum import Enum
import subprocess
from functools import partial
import numpy as np
import os
from Agent import Agent

## Feature

MAP = np.array([])
SNAKE_HEAD_COORDINATE = (-1, -1)
APPLE_COORDINATE = (-1, -1)
SCOURE = -3


class windowNotFoundError(Exception):
    def __init__(self, message="windowNotFound !!"):
        self.message = message
        super().__init__(self.message)

class REWARD(Enum):
    EAT_APPLE = 1000
    MOVE = -1
    LOSE = -2000

class STATE(Enum):
    MOVE = 0,
    LOSE = 2,
    EAT_APPLE = 3,

class WORLD(Enum):
    EMPTY_CELL = 0
    WALL = 1
    APPLE = 2
    SNAKE_BODY_CELL = 3
    SNAKE_HEAD = 4



    
def get_window():
    display_obj = display.Display()
    root = display_obj.screen().root
    windowIDs = root.get_full_property(display_obj.intern_atom('_NET_CLIENT_LIST'), X.AnyPropertyType).value
    for windowID in windowIDs:
            window = display_obj.create_resource_object('window', windowID)
            window_name = window.get_full_property(display_obj.intern_atom('_NET_WM_NAME'), X.AnyPropertyType).value
            if window_name == "SNAKE_GAME_WINDOWS":
               return window
    
    raise windowNotFoundError()
    


def get_window_image(window):
    geometry = window.get_geometry()
    raw = window.get_image(0, 0, geometry.width, geometry.height, X.ZPixmap, 0xffffffff)
    image = Image.frombytes("RGB", (geometry.width, geometry.height), raw.data, "raw", "BGRX")
    return image


def get_window_pixle_map(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixel_map = image.load()
    return pixel_map

def get_cell_value(pixle_map, x, y):
    r_sum = 0
    g_sum = 0
    b_sum = 0
    for i in range(20):
            pixel = pixle_map[((y*20) + i), ((x*20) + i)]
            r, g, b = pixel
            r_sum += r
            g_sum += g
            b_sum += b
        
    r_mean = r_sum / 21
    g_mean = g_sum / 21
    b_mean = b_sum / 21
            
            
    if g_mean > 100 and r_mean > 100:
        return WORLD.SNAKE_HEAD.value
    elif r_mean > 100:
        return WORLD.APPLE.value
    elif b_mean > 100:
        return WORLD.WALL.value
    elif g_mean > 100:
        return WORLD.SNAKE_BODY_CELL.value
    return WORLD.EMPTY_CELL.value
 

def update_features(pixle_map, image):
    snake_head = [-1, -1]
    apple = [-1, -1]
    scoure = -3 ##body_length
    
    map = np.zeros((int(image.height/20), int(image.width/20)) , dtype=np.int32)
    print(map.shape)
    for (x, row)in enumerate(map):
        for (y, cell) in enumerate(row):
            value = get_cell_value(pixle_map, x, y)
            if value == WORLD.SNAKE_HEAD.value:
                snake_head[0]=x
                snake_head[1]=y
                scoure+=1
                
            elif value == WORLD.APPLE.value:
                apple[0]=x
                apple[0]=y
                
            elif value == WORLD.SNAKE_BODY_CELL.value:
                scoure+=1
                
            map[x][y] = value

    return map, snake_head, apple, scoure 
             
            
            
def update(window):
    image = get_window_image(window=window)
    pixle_map = get_window_pixle_map(image=image)
    MAP, SNAKE_HEAD_COORDINATE, APPLE_COORDINATE, SCOURE = update_features(pixle_map, image)
    
    os.system("cls" if os.name == "nt" else "clear")
    for row in MAP:
        for cell in row:
            print(cell, end=" ")
        print()


############# main ###############
if __name__ == "__main__":
    process = subprocess.Popen(['./build/snake_game'], close_fds=True)
    time.sleep(2)
    try:
        window = get_window()
        update(window=window)
        process.terminate()
        
    except windowNotFoundError as e:
        print(e)