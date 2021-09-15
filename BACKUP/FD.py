from gpiozero import Robot
from gpiozero import Servo
from time import sleep
robot = Robot(left=(20,26), right=(18, 27))
rf= Robot(left=(8,7), right=(16,19))
servo=Servo(14)
servo.min()
try:
    
   while True:
       var=input()
       if(var=="w"):
           robot.forward(0.25)
           rf.forward(0.25)
#                sleep(1)
#            robot.stop()
#            rf.stop()
       elif(var=='s'):
            servo.max()
        
       elif(var=='e'):
            servo.mid()
       elif(var=='a'):
           robot.left(0.25)
           rf.left(0.25)
#            sleep(1)
#            robot.stop()
#            rf.stop()
           print("LEFT")
       elif(var=='d'):
           robot.right(0.25)
           rf.right(0.25)
#            sleep(1)
#            robot.stop()
#            rf.stop()
           print("RIGHT")
       elif(var=='q'):
            robot.stop()
            rf.stop()
except KeyboardInterrupt:
       pass