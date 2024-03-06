import pyfirmata
import time

# change based on your port
board = pyfirmata.ArduinoNano('COM6')

while True:

    board.digital[13].write(1)

    time.sleep(1)

    board.digital[13].write(0)

    time.sleep(1)