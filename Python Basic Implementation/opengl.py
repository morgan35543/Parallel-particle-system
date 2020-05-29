import sys 
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
from particleCode import *
import math

# define size of image
width, height = 512, 512

listofparticles = initialiseArray()
atomic = AtomicCounter()
gravityhalt = False
solidcolour = False # Key 1
brightnessspeed = False # Key 2
centermass = False # Key 4
rendertext = ""

def draw():
    global image, width, height, listofparticles
    particlecolour = 0 # Red by default

    if gravityhalt == True:
        listofparticles = gravstopmethod(listofparticles)
        particlecolour = 2 # Blue
    else:
        listofparticles = mainmovementandcollisions(atomic, listofparticles)

    for particle in listofparticles:
        particleX = particle._x
        particleY = particle._y
        particlebrightness = 255

        if solidcolour == True:
            particlecolour = 1 # Green

        if centermass == True:
            centerX = height / 2
            centerY = width / 2
            tempX = 0
            tempY = 0
            radius = height / 2
            additionalradius = 30 # allows center of mass to be further out


            tempX = centerX - particleX
            tempY = centerY - particleY

            hypotenuse = (tempX * tempX) + (tempY * tempY)
            hypotenuse = math.sqrt(hypotenuse)

            if hypotenuse > (radius - additionalradius):                   
                #print("continue")
                continue

            particlebrightness = 255 - (hypotenuse + additionalradius)    
            
        if brightnessspeed == True:
            particlecolour = 1 # Green
            if particle._vx and particle._vy == 1:
                particlebrightness = 120
            elif particle._vx and particle._vy == 2: 
                particlebrightness = 180
            elif particle._vx and particle._vy == 3:
                particlebrightness = 255
            elif particle._vx and particle._vy == 0:
                particlebrightness = 75
            elif particle._vx or particle._vy == 3:
                particlebrightness = 220
            elif particle._vx or particle._vy == 2:
                particlebrightness = 150
            elif particle._vx or particle._vy == 1:
                particlebrightness = 90

        image[particleX][particleY][particlecolour] = particlebrightness

def render_string(text: str, x: int, y: int):
    gl.glColor3f(1, 1, 1)
    gl.glRasterPos2f(x,y)

    for i in range(len(text)):
        glut.glutBitmapCharacter(gl.OpenGL.GLUT.fonts.GLUT_BITMAP_HELVETICA_18, ord(text[i]))

def displayCallback():    
    start_time = time.time()
    global image, width, height, listofparticles
    image = np.zeros((width, height, 4), dtype=np.ubyte)

    draw()
    
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glLoadIdentity()

    # Draw image
    gl.glRasterPos2i(-1, -1)
    gl.glDrawPixels(width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image)
    render_string(rendertext, 0, 0)
    glut.glutSwapBuffers()
    print("FPS: ", 1.0 / (time.time() - start_time))

def reshapeCallback(width, height):
    gl.glClearColor(1, 1, 1, 1)
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

def keyboardCallback(key, x, y):
    global gravityhalt, solidcolour, brightnessspeed, centermass, rendertext
    
    if key == b'\033':
      sys.exit( )
    elif key == b'q':
      sys.exit( )   
    elif key == b'g':
        if gravityhalt == False:
            rendertext = "Gravity Engaged"
            gravityhalt = True
    elif key == b'1':
        if solidcolour == False:            
            rendertext = "1 - Solid Colour"
            solidcolour = True
            brightnessspeed = False
            centermass = False
        elif solidcolour == True:
            rendertext = ""
            solidcolour = False
            brightnessspeed = False
            centermass = False
    elif key == b'2':
        if brightnessspeed == False:            
            rendertext = "2 - Brightness Speed"
            solidcolour = False
            brightnessspeed = True
            centermass = False
        elif brightnessspeed == True:
            rendertext = ""
            solidcolour = False
            brightnessspeed = False
            centermass = False
    elif key == b'3':        
        rendertext = "Sorry key 3 not implemented"
    elif key == b'4':
        if centermass == False:
            rendertext = "4 - Center Mass"
            solidcolour = False
            brightnessspeed = False
            centermass = True
        elif centermass == True:
            rendertext = ""
            solidcolour = False
            brightnessspeed = False
            centermass = False

def keyboardUpCallback(key, x, y):
    global gravityhalt, rendertext
    if key == b'g':
        if gravityhalt == True:
            rendertext = ""
            gravityhalt = False
    elif key == b'3':        
        rendertext = ""



if __name__ == "__main__":
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | 
    glut.GLUT_DEPTH)
    glut.glutInitWindowSize(512, 512)
    glut.glutInitWindowPosition(704, 259)
    glut.glutCreateWindow('Python particles simulator')
    glut.glutDisplayFunc(displayCallback)
    glut.glutIdleFunc(displayCallback)
    glut.glutReshapeFunc(reshapeCallback)
    glut.glutKeyboardFunc(keyboardCallback)
    glut.glutKeyboardUpFunc(keyboardUpCallback)

    #   Create image
    image = np.zeros((width, height, 4), dtype=np.ubyte)
    glut.glutMainLoop()