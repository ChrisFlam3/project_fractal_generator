import sfml.sf as sf


def main():
    window = sf.RenderWindow(sf.VideoMode(1920 , 1080), 'Drawing an image with SFML')
    window.framerate_limit = 60
    running = True
    texture = sf.Texture.from_file('1.jpg')
    sprite = sf.Sprite(texture)
    view=sf.View(sf.Rect((0,0),(500,500)))
    window.view=view
    while running:
        for event in window.events:
            if event.type == sf.Event.CLOSED:
                running = False

        if sf.Keyboard.is_key_pressed(sf.Keyboard.RIGHT):
            sprite.move(5, 0)
        elif sf.Keyboard.is_key_pressed(sf.Keyboard.LEFT):
            sprite.move(-5, 0)

        window.clear(sf.Color.WHITE)
        window.draw(sprite)
        window.display()

    window.close()


if __name__ == '__main__':
    main()