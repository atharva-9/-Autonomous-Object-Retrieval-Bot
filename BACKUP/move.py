from gpiozero import Robot
robot = Robot(left=(4, 14), right=(17, 27))
# 
# from gpiozero import DistanceSensor
# from time import sleep
# 
# from gpiozero import AngularServo
# s = AngularServo(20, min_angle=-42, max_angle=44)
# 
# sensor = DistanceSensor(echo=12, trigger=16)
# 
# while True:
#     print('Distance: ', sensor.distance * 100)
# 
# if(sensor.distance*100>20):
#     robot.stop()
#     s.angle=0
# else :            
#     if((4.99>degs>0.0)or(degs>=85)):
#         robot.forward(0.5)
#         print("FORWARD")
#     elif((50.0>degs)and(degs>=5.0)):
#         robot.right(0.5)
#         print("RIGHT")
#     elif((50.0<degs)and (degs<=85.0)):
#         robot.left(0.5)
#         print("LEFT")
#     else :
#         robot.stop()
#         s.max()
from gpiozero import AngularServo
from time import sleep
servo=AngularServo(14, min_angle=-42, max_angle=180)
servo.angle=180
#servo.min()
# sleep(5)
# servo.min()
# sleep(1)
