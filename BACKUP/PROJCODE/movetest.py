from gpiozero import Robot
from time import sleep

robot = Robot(left=(4, 14), right=(17, 18))
robot.forward()
sleep(1)
robot.right()
sleep(1)
robot.left()
sleep(1)
robot.stop()
print('Done')