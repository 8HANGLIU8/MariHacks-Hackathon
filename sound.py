
#source my_music_player/bin/activate

import pygame
import time
#from Felix import direction as x, distance as n

pygame.init()
pygame.mixer.init()
clock = pygame.time.Clock()

# Création de la fenêtre
gameDisplay = pygame.display.set_mode((800, 800))

# Direction actuelle
x = "left"  # Peut être "left", "right", ou "straight"
n = 1 # 
# Fréquences pour chaque direction (en secondes entre les sons)
frequencies = {
    "left": n,      # joue toutes les 1 seconde
    "right": n,     # joue toutes les 2 secondes
    "straight": n   # joue toutes les 3 secondes
}

# Temps du dernier son joué
last_played_time = 0

def play_direction_sound(direction):
    if direction == "left":
        pygame.mixer.music.load("left.mp3")
    elif direction == "right":
        pygame.mixer.music.load("right.mp3")
    elif direction == "straight":
        pygame.mixer.music.load("straight.mp3")
    else:
        return
    pygame.mixer.music.play(0)

# Boucle principale
running = True
while running:
    pygame.display.update()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

            #from Felix import direction as x, distance as n

        # Changer de direction avec les touches fléchées
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x = "left"
            elif event.key == pygame.K_RIGHT:
                x = "right"
            elif event.key == pygame.K_UP:
                x = "straight"

    # Jouer le son selon la fréquence
    current_time = time.time()
    frequency = frequencies.get(x, 2.0)  # par défaut: 2s

    if current_time - last_played_time >= frequency:
        play_direction_sound(x)
        last_played_time = current_time

    clock.tick(60)

pygame.quit()