
#source my_music_player/bin/activate

import pygame

#from felix import direction, distance
pygame.init()
pygame.mixer.init()
clock = pygame.time.Clock()

gameDisplay = pygame.display.set_mode((800,800))
while True:
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            pygame.mixer.music.load("right.mp3")
            pygame.mixer.music.play(-1)
            pygame.mixer.music.play(0) 
    pygame.mixer.init()
    clock.tick(60)
    

x = "right"
if x == "right":
    print("Right")
    pygame.mixer.init()
    pygame.mixer.music.load("right.mp3")
    pygame.mixer.music.play(-1)
elif x == "left":
    print("Left")
    pygame.mixer.init()
    pygame.mixer.music.load("left.mp3")
    pygame.mixer.music.play("left.mp3")
elif x == "straight":
    print("Straight")
    pygame.mixer.init()
    pygame.mixer.music.load("straight.mp3")
    pygame.mixer.music.play("straight.mp3")

pygame.quit()
"""
while True:

    x = 1 #direction()
    y = 1 #distance()

    if x:
        if x == "right":
            print("Right")
            pygame.mixer.init()
            pygame.mixer.music.load("right.mp3")
            pygame.mixer.music.play("right.mp3")
        elif x == "left":
            print("Left")
            pygame.mixer.init()
            pygame.mixer.music.load("left.mp3")
            pygame.mixer.music.play("left.mp3")
        elif x == "straight":
            print("Straight")
            pygame.mixer.init()
            pygame.mixer.music.load("straight.mp3")
            pygame.mixer.music.play("straight.mp3")
    else:
        print("No direction received. Doing nothing.")

    if y:
        frequency = 1000
        result = frequency / y
        print(f"Result of frequency / distance: {result}")
        pygame.mixer.init()
        pygame.mixer.music.load("bip.mp3")
        pygame.mixer.music.play("bip.mp3")
        
    else:
        print("No distance received. Doing nothing.")

"""