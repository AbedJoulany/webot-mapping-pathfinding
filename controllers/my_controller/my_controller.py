#from controller import Robot, Keyboard, DistanceSensor, Motor
from controller import Robot, Keyboard, GPS, InertialUnit, Motor

TIME_STEP = 64


robot = Robot()
kb = robot.getKeyboard()
gps = robot.getDevice('global')
imu = robot.getDevice('imu')
lr = robot.getDevice('linear')
rm = robot.getDevice('RM')
cm = robot.getDevice('camera')



ds = []
dsNames = ['ds_right', 'ds_left']

for i in range(2):
    ds.append(robot.getDevice(dsNames[i]))
    ds[i].enable(TIME_STEP)



# Uncomment these lines to enable GPS and InertialUnit
# gp = robot.getGPS("global")
# gp.enable(TIME_STEP)

# iu = robot.getInertialUnit("imu")
# iu.enable(TIME_STEP)

    
wheels = []
wheelsNames = ['fr_left_wheel', 'fr_right_wheel', 'bk_left_wheel',  'bk_right_wheel']

for i in range(4):
    wheels.append(robot.getDevice(wheelsNames[i]))
    wheels[i].setPosition(float('inf'))
    wheels[i].setVelocity(0.0)


kb.enable(TIME_STEP)
gps.enable(TIME_STEP)
imu.enable(TIME_STEP)
cm.enable(TIME_STEP)


leftSpeed = 0.0
rightSpeed = 0.0
linear = 0
rotate = 0

while robot.step(TIME_STEP) != -1:
    key = kb.getKey()
    
    if key == Keyboard.UP:
        leftSpeed = 1.0
        rightSpeed = 1.0
    elif key == Keyboard.DOWN:
        leftSpeed = -1.0
        rightSpeed = -1.0
    elif key == Keyboard.RIGHT:
        leftSpeed = 1.0
        rightSpeed = -1.0
    elif key == Keyboard.LEFT:
        leftSpeed = -1.0
        rightSpeed = 1.0
    else:
        leftSpeed = 0.0
        rightSpeed = 0.0
    
    print(f"{ds[0].getValue()} = Right Sensor")
    print(f"{ds[1].getValue()} = Left Sensor")
    wheels[0].setVelocity(leftSpeed)
    wheels[1].setVelocity(rightSpeed)
    wheels[2].setVelocity(leftSpeed)
    wheels[3].setVelocity(rightSpeed)
    gps_values = gps.getValues()
    print(f"X : {gps_values[0]}")
    print(f"Y : {gps_values[1]}")
    print(f"Z : {gps_values[2]}")
    print("########################")
    
    imu_values = imu.getRollPitchYaw()
    
    print(f"Angle X : {imu_values[0]}")
    print(f"Angle Y : {imu_values[1]}")
    print(f"Angle Z : {imu_values[2]}")
    
    
    if key == 87 and linear < 0.19:
        linear+=0.005
    elif key == 83 and linear > 0:
        linear-=0.005
    else:
        linear +=0
        
    lr.setPosition(linear)
     
    if key == 65 and rotate < 1.57:
        rotate+=0.05
    elif key == 68 and rotate > -1.57:
        rotate-=0.05
    else:
        rotate +=0
    rm.setPosition(rotate)


