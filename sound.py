
#source my_music_player/bin/activate


import pygame

pygame.init()
pygame.mixer.init()
clock = pygame.time.Clock()

# Création de la fenêtre
gameDisplay = pygame.display.set_mode((800, 800))

# Fonction pour jouer le son selon la direction
def play_direction_sound(x):
    if x == "left":
        pygame.mixer.music.load("left.mp3")
    elif x == "right":
        pygame.mixer.music.load("right.mp3")
    elif x == "straight":
        pygame.mixer.music.load("straight.mp3")
    else:
        return  # Ne rien faire si la direction n'est pas reconnue
    pygame.mixer.music.play(0)

# Exemple de direction
x = "left"  # <- Tu peux changer cette valeur pour tester d'autres directions

# Boucle principale
running = True
while running:
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            play_direction_sound(x)

    clock.tick(60)

pygame.quit()
