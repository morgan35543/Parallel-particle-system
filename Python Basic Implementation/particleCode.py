import logging
import time
import threading
from collections import Counter
from collections import defaultdict
import numpy as np

class Particle(object):
    def __init__(self):
        self._x = 0
        self._vx = 0
        self._y = 0
        self._vy = 0
        self._lock = threading.Lock()

    def __iter__(self):
        yield self._x, self._y

    def setPosition(self, x, y):       
        with self._lock:
            self._x = x
            self._y = y
        
    def setVelocities(self, vx, vy):
        with self._lock:
            self._vx = vx
            self._vy = vy
        
class AtomicCounter:
    def __init__(self, value=0):
        self._value = value
        self._lock = threading.Lock()

    def inc(self):
        with self._lock:
            self._value += 1
            return self._value

    def dec(self):
        with self._lock:
            self._value -= 1
            return self._value

    @property
    def value(self):
        with self._lock:
            return self._value

    @value.setter
    def value(self, v):
        with self._lock:
            self._value = v
            return self._value

def initialPosition(particle, listofpositions, boundary):  
    position = [particle._x, particle._y]

    updateX = position[0] 
    updateY = position[1]
    position = [updateX, updateY] 

    while position in listofpositions:
        logging.info("Main : Position taken") 
        if updateX > boundary:
            updateX = updateX - boundary

        if updateY > boundary:
            updateY = updateY - boundary                
        
        updateX = position[0] + 1
        updateY = position[1] + 1
        position = [updateX, updateY]        
        
    return position

def updatePosition(index, listofparticles, boundary):  
    # Summary:
    # Updates particle positions using their internal velocities.
    # This method works on individual threads.
    # This method runs on a seperate thread for each particle.
    # Each Particle() is locked when writing to therefore threadsafe.
    # If a particle exceeds/touches a boundary, it's velocity is reversed
    
    particle = listofparticles[index]
    position = [particle._x, particle._y]

    updateX = particle._x + particle._vx
    updateY = particle._y + particle._vy

    if updateX > boundary: 
        particle._vx = -particle._vx       
        updateX = boundary - (updateX - boundary)
    elif updateX < 0:
        particle._vx = -particle._vx       
        updateX = -updateX
    elif updateX == boundary or updateX == 0:
        particle._vx = -particle._vx

    if updateY > boundary:
        particle._vy = -particle._vy      
        updateY = boundary - (updateY - boundary)
    elif updateY < 0:
        particle._vy = -particle._vy        
        updateY = -updateY
    elif updateY == boundary or updateY == 0:
        particle._vy = -particle._vy  
       
    position = [updateX, updateY] 
    particle.setPosition(updateX, updateY)
    return position

# Stops X-axis movement. All particles fall down y-Axis 1 point per loop
# To make particles fall through change fall-through to true (Manually)
def haltforgravity(index, listofparticles, boundary):
    particle = listofparticles[index]
    
    # In the openGL x,y are reversed 
    updateX = particle._x
    updateY = particle._y
   
    fallthrough = True

    if updateX > 0:
        updateX = particle._x - 1
    elif fallthrough == True and updateX == 0:
        updateX = boundary

    particle.setPosition(updateX, updateY)

# Each collision is its own thread
# May be slow if multiple particles collide in the same space
numofcollisons = 0
def checkPossibleCollisions(particlesindices, listofparticles):
    velocities = []
    positions = []
    for index in particlesindices:
        positions.append([listofparticles[index]._x, listofparticles[index]._y])
        velocities.append([listofparticles[index]._vx, listofparticles[index]._vy])
    
    # Calculate which axis the particles intersect on and reverse it (Basically swap the velocities on the axis they hit on)
    # Prioritise converting momentum to the stationary axis' first
    #print(len(velocities))
    #print("Before:",velocities)
    if len(velocities) == 2:
        vel1 = velocities[0]
        vel2 = velocities[1]

        if vel1[0] == -vel2[0] or -vel1[0] == vel2[0]:
            vel1[0] = -vel1[0]
            vel2[0] = -vel2[0]
        elif vel1[0] == 0 and vel2[0] != 0:
            vel1[0] = vel2[0]
            vel2[0] = 0
        elif vel1[0] != 0 and vel2[0] == 0:
            vel2[0] = vel1[0]
            vel1[0] = 0
        
        if vel1[1] == -vel2[1] or -vel1[1] == vel2[1]:
            vel1[1] = -vel1[1]
            vel2[1] = -vel2[1]
        elif vel1[1] == 0 and vel2[1] != 0:
            vel1[1] = vel2[1]
            vel2[1] = 0
        elif vel1[1] != 0 and vel2[1] == 0:
            vel2[1] = vel1[1]
            vel1[1] = 0

        velocities = [vel1, vel2]
    else:
        pass

    #print("After:",velocities)
    counter = 0
    for index in particlesindices:
        velocity = velocities[counter]
        counter += 1
        listofparticles[index].setVelocities(velocity[0], velocity[1])
    #print(positions)
    #time.sleep(5)
      
lock = threading.Lock()
def thread_main(threadId, listofparticles, boundary):
    global lock  
    #logging.info("Movement Thread %d : Begin",id)
    with lock:
        updatePosition(threadId, listofparticles, boundary)
    #logging.info("Movement Thread %d : End",id)

def thread_collision(particlesindices, listofparticles):
    global lock  
    #logging.info("Collision Thread %d : Begin",id)
    with lock:
        checkPossibleCollisions(particlesindices, listofparticles)
    #logging.info("Collision Thread %d : End",id)

# Each particle has it's own thread when gravity in effect
# Should take over from thread_main and return control afterwards
# Internal vectors remain unaffected
def thread_gravstop(threadId, listofparticles, boundary):
    global lock  
    #logging.info("Gravity Thread %d : Begin",id)
    with lock:
        haltforgravity(threadId, listofparticles, boundary)
    #logging.info("Gravity Thread %d : End",id)

# [x,y,z] lengths - Positives only
boundary = 511
numberofparticles = 1000
def initialiseArray():
    # Initialising particles
    # Each partical is placed in a unique space [x,y,z]

    # Particle movement speed limit
    # When a particle is initialised each vector is given a random velocity between -speedLimit and speedLimit
    # This allows for random movement with particles having different velocities on different vectors
    speedLimit = 3

    listofparticles = []
    listofpositions = []
    global numberofparticles
    global boundary
    for particleId in range(numberofparticles):
        arrayBoundary = boundary - 1        
        randomX = np.random.randint(0, arrayBoundary)
        randomY = np.random.randint(0, arrayBoundary)
        position = [randomX, randomY]
        
        randomVX = np.random.randint(-speedLimit, speedLimit)
        randomVY = np.random.randint(-speedLimit, speedLimit)
        while randomVX == 0 and randomVY == 0:
            randomVX = np.random.randint(-speedLimit, speedLimit)
            randomVY = np.random.randint(-speedLimit, speedLimit)

        velocities = [randomVX, randomVY]
        particle = Particle()
        particle.setPosition(position[0], position[1])
        particle.setVelocities(velocities[0], velocities[1])

        if [particle._x, particle._y] in listofpositions: 
            logging.info("Main : Finding new")       
            newPos = initialPosition(particle, listofpositions, boundary)
            logging.info("Main : New obtained")      
            listofpositions.append(newPos)
            logging.info("Main : Position set")      
            particle.setPosition(newPos[0], newPos[1])
            logging.info("Main : Particle set")      
            listofparticles.append(particle)
            logging.info("Main : Particle Added") 
        else: 
            logging.info("Main : Position set")   
            #print(position)
            listofpositions.append([particle._x, particle._y])
            listofparticles.append(particle)     
    return listofparticles

counter = 0
def mainmovementandcollisions(atomic, listofparticles):
    global numberofparticles
    global boundary
    global counter
    numberofparticlethreads = numberofparticles
    
    logging.info("Main : Position/Collision loop: %d", counter)
    counter += 1
    # Obtains the current state of each object placing into a list of objects
    particlepositions = []
    for particle in listofparticles:
        particlepositions.append([particle._x, particle._y])

    # Places each object into a thread where it's vectors are updated based on the object velocity
    listofparticlethreads=[] 
    velocitylist = []
    for id in range(numberofparticlethreads):
        velocities = [listofparticles[id]._vx, listofparticles[id]._vy]
        velocitylist.append(velocities)
        particlethread = threading.Thread(target=thread_main,args=(id, listofparticles, boundary))
        listofparticlethreads.append(particlethread)
        particlethread.start()

    # Joins the position update threads
    for particlethread in listofparticlethreads:
        particlethread.join()

    # Debugging positions & velocities - Manually change boolean
    debugBool = False
    if debugBool:
        print(particlepositions)
        #print(velocitylist)
        sleeptimer = 2
        logging.info("Main : Sleeping for %d seconds",sleeptimer)
        time.sleep(sleeptimer)

    # Colliding particles only register if they occupy the same space
    # A list of indices that are occupied by two or more particles is returned
    # This is done to reduce the search space and increase algorithm performance
    tupleDups = Counter(map(tuple,listofparticles))
    dups = [k for k,v in tupleDups.items() if v > 1] 
    dupList = []
    if len(dups) > 0:
        for dup in dups:
            # Currently tuple within tuple, retrieves the actual coordiantes tuple
            position = dup[0]
            # Converts coordinates tuple to a list
            positionaslist = [position[0], position[1]]
            dupList.append(positionaslist)

    # Turns unhashable lists into hashable tuples
    particlepositions = []
    for particle in listofparticles:
        particlepositions.append(tuple([particle._x, particle._y]))

    # Enumerate the tuples storing them and the particle indices in a dict
    d = defaultdict(list)
    for keyindex, value in enumerate(particlepositions):
        d[value].append(keyindex)
        
    collisionthreads = []
    # Retrieve duplicate indices (Particles that collide this iteration)
    for dup in dupList:
        dup = tuple(dup)
        collisionthread = threading.Thread(target=(checkPossibleCollisions), args=(d[dup], listofparticles))
        collisionthread.start()
        collisionthreads.append(collisionthread)
        atomic.inc()
        #print(d[dup])
        
    for collisionthread in collisionthreads:
        collisionthread.join()
        
    particlepositions = []
    for particle in listofparticles:
        particlepositions.append(tuple([particle._x, particle._y]))

    # Debugging positions & velocities - Manually change boolean
    debugBool = False
    if debugBool:
        print(particlepositions)
        #print(velocitylist)
        sleeptimer = 2
        #logging.info("Main : Sleeping for %d seconds",sleeptimer)
        #time.sleep(sleeptimer)

    return listofparticles

def gravstopmethod(listofparticles):
    global numberofparticles
    gravstopthreads = []
    for id in range(numberofparticles):
        gravitythread = threading.Thread(target=(thread_gravstop), args=(id, listofparticles, boundary))
        gravstopthreads.append(gravitythread)
        gravitythread.start()

    for gravitythread in gravstopthreads:
        gravitythread.join()

    particlepositions = []
    for particle in listofparticles:
        particlepositions.append([particle._x, particle._y])
    
    #print(particlepositions)
    #time.sleep(1)
    return listofparticles

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s: %(message)s",level=logging.INFO,datefmt="%H:%M:%S")    
    logging.info("Main : Begin")
    
    # Initialise array
    logging.info("Main : Particles initialising...")
    listofparticles = initialiseArray()
    logging.info("Main : Particles initialised.")
    
    # Log to show no duplicates exist initially
    # Not necessary in openGL
    logging.info("Main : Checking for duplicate initial positions")
    particlepositions = []
    for particle in listofparticles:
        particlepositions.append([particle._x, particle._y])
    c = Counter(map(tuple,particlepositions))
    dups = [k for k,v in c.items() if v>1]  
    dupCount = len(dups)    
    if dupCount > 0:
        print("Number of duplicates found:", dupCount)
        logging.info("Main : Printing duplicate positions")
        print(dups)
        sleeptimer = 10
        logging.info("Main : Sleeping for %d seconds", sleeptimer)
        time.sleep(sleeptimer)
    else:        
        logging.info("Main : No duplicate positions : Simulation starting...")

    gravstop = True
    atomic = AtomicCounter()
    timeout = time.time() + (60 * 2) 
    gravtimer = 0
    while True:
        listofparticles = mainmovementandcollisions(atomic, listofparticles)  
        
        enteredGrav = False
        while gravtimer > 10 and gravtimer < 25:
            if enteredGrav == False:
                logging.info("Main : Gravity in effect")

            enteredGrav = True
            listofparticles = gravstopmethod(listofparticles)

            #gravstopthreads = []
            #for id in range(numberofparticles):
            #        gravitythread = threading.Thread(target=(thread_gravstop), args=(id, listofparticles, boundary))
            #        gravstopthreads.append(gravitythread)
            #        gravitythread.start()

            #for gravitythread in gravstopthreads:
            #    gravitythread.join()

            #particlepositions = []
            #for particle in listofparticles:
            #    particlepositions.append([particle._x, particle._y])

            #print(particlepositions)
            #time.sleep(1)
            gravtimer += 1

        if enteredGrav == True:
            logging.info("Main : Gravity no longer effect")

        #time.sleep(1)
        gravtimer += 1

        if time.time() > timeout:
            break
        
    print("Total collisions:", atomic.value)
    logging.info("Main : End")