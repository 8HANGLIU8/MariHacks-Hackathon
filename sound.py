from felix import direction, distance

while True:
    x = direction()
    y = distance()
    frequency = 100

    if x:
        if x == "right":
            print("Right")
        elif x == "left":
            print("Left")
        elif x == "straight":
            print("Straight")
    else:
        print("No direction received. Doing nothing.")

    if y:
        
        result = frequency / y
        print(f"Result of frequency / distance: {result}")
    else:
        print("No distance received. Doing nothing.")