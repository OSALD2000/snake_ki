from Xlib import X, display
from PIL import Image
import time
from enum import Enum
import subprocess
import numpy as np
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
    MOVE = 0
    LOSE = 2
    EAT_APPLE = 3

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
               return window , windowID
    
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
    snake_head = np.array([-1, -1])
    apple = np.array([-1, -1])
    scoure = -3 ##body_length
    
    map = np.zeros((int(image.height/20), int(image.width/20)) , dtype=np.int32)
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
    return update_features(pixle_map, image)



def calculate_new_state(window):
    map, NEW_SNAKE_HEAD_COORDINATE, apple, NEW_SCOURE = update(window=window)
    
    if NEW_SCOURE == 0 and ((SNAKE_HEAD_COORDINATE[0] - NEW_SNAKE_HEAD_COORDINATE[0])**2) > 1 and ((SNAKE_HEAD_COORDINATE[1] - NEW_SNAKE_HEAD_COORDINATE[1])**2) > 1:
        return STATE.LOSE.value, REWARD.LOSE.value
    
    if NEW_SCOURE > SCOURE:
        return STATE.EAT_APPLE.value, REWARD.EAT_APPLE.value
    
    return STATE.MOVE.value , REWARD.MOVE.value

def get_env(window):
        
        map, snake_head, apple, scoure = update(window=window)
        map_flat = map.flatten()
        snake_head_flat = snake_head.flatten()
        apple_flat = apple.flatten()

        combined_flat = np.concatenate([map_flat, snake_head_flat, apple_flat])

        env = combined_flat.reshape(1, -1)
        
        return env
    
############# main ###############
        
if __name__ == "__main__":
    try:
        
        num_episodes = 1000
        max_steps_per_episode = 1000
        learning_rate = 0.1
        discount_rate = 0.99
        exploration_rate = 1
        max_exploration_rate = 1
        min_exploration_rate = 0.01
        exploration_decay_rate = 0.01
        
        window, window_id = get_window()
        MAP, SNAKE_HEAD_COORDINATE, APPLE_COORDINATE, SCOURE = update(window=window)
        map_flat = MAP.flatten()
        snake_head_flat = SNAKE_HEAD_COORDINATE.flatten()
        apple_flat = APPLE_COORDINATE.flatten()
        
        combined_flat = np.concatenate([map_flat, snake_head_flat, apple_flat])
        
        feature_size = combined_flat.shape[0]
        
        agent = Agent(state_size=3, action_size=4, feature_size = feature_size)
        
        for episode in range(num_episodes):
            state = STATE.MOVE.value
            total_reward = 0
            
            for step in range(max_steps_per_episode):

                old_env = get_env(window)

                action = agent.act(env=[old_env])
                    
                new_state, reward = calculate_new_state(window)
                
                new_env = get_env(window)
                
                agent.update_q_values([old_env], action, reward, [new_env], learning_rate, discount_rate)
                
                total_reward += reward
                state = new_state
                
                if state == STATE.LOSE.value:
                    break
            
            
            print("Episode:", episode, "Total Reward:", total_reward)
            
    except Exception as e:
        print("Error:", e)
