"""
File: motors.py
Author: CompSci Tech Team (sci.compsci.tech@mailbox.lboro.ac.uk)
Date: 28/01/2025
Description: Python class that passes messages based upon methods to the Yukon via serial to control the Teaching Robots V2 motors. 
"""

import serial
import time

class MotorsYukon():
    def __init__(self,port="/dev/ttyTHS1", mecanum=False):
        """
        In
        The robt can work with noral or mecanum wheels,
        """
        self.port=port
        self.baudrate=115200
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.mecanum=mecanum
        return

    def connect(self):
        # Open the connection to Yukon via UART 
        self.serial_port = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
        )
        return
    
    def close(self):
        # Connect to Yukon via UART 
        self.serial_port.close()
        return

    def send(self, message):
        #Connect - send - message and close serial
        self.connect()
        self.serial_port.write(message.encode())
        self.close()
        return

    def forward(self, speed=0.3):
        # Move the robot forwards 
        message=f"m;direction:forward;speed:{speed}\n"
        self.send(message)
        return

    def backward(self, speed=0.3):
        message = f"m;direction:backward;speed:{speed}\n"
        self.send(message)
        return

    def right(self, speed=0.3):
        #If mecanum wheel move robot left else spin left
        if self.mecanum:
            message = f"m;direction:right;speed:{speed}\n"
        else:
            message = f"m;direction:spinright;speed:{speed}\n"
        self.send(message)
        return

    def left(self, speed=0.3):
        if self.mecanum:
            message = f"m;direction:left;speed:{speed}\n"
        else: 
            message = f"m;direction:spinleft;speed:{speed}\n" 
        self.send(message)
        return
    
    def spinRight(self, speed=0.3):
        message = f"m;direction:spinright;speed:{speed}\n"
        self.send(message)
        return

    def spinLeft(self, speed=0.3):
        message = f"m;direction:spinleft;speed:{speed}\n"
        self.send(message)
        return
    
    def stop(self):
        speed=0.0
        message = f"m;direction:stop;speed:{speed}\n"
        self.send(message)
        return

    #Single motor methods 
    def frontLeft(self, speed=0.0):
        message = f"m;direction:frontleft;speed:{speed}\n"
        self.send(message)
        return
    
    def frontRight(self, speed=0.0):
        message = f"m;direction:frontright;speed:{speed}\n"
        self.send(message)
        return
    
    def backLeft(self, speed=0.0):
        message = f"m;direction:backleft;speed:{speed}\n"
        self.send(message)
        return
    
    def backRight(self, speed=0.0):
        message = f"m;direction:backright;speed:{speed}\n"
        self.send(message)
        return
    
if __name__ == "__main__":
    motorTest = MotorsYukon(mecanum=True)
    print("Forward")
    motorTest.left()
    time.sleep(2)
    print("Stop")
    motorTest.stop()
