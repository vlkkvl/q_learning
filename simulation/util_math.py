def screen_to_list(screen_x, screen_y):
    x = int((screen_x + 288) / 24)
    y = int((288 - screen_y) / 24)
    return x, y

def list_to_screen(x, y):
    screen_x = -288 + (x * 24)
    screen_y =  288 - (y * 24)
    return screen_x, screen_y