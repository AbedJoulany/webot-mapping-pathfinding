from controller import  Supervisor, Keyboard, Lidar, GPS

class RobotController:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._robot = Supervisor()
            cls._instance._timestep = int(cls._instance._robot.getBasicTimeStep())
            cls._instance._initialize_devices()
        return cls._instance

    def _initialize_devices(self):
        self._lidar = self._robot.getDevice("lidar")
        self._lidar.enable(self._timestep)
        self._lidar.enablePointCloud()
        
        self._gps = self._robot.getDevice("gps")
        self._gps.enable(self._timestep)
        
        # Compass
        self._compass = self._robot.getDevice("compass")
        self._compass.enable(self._timestep)

        self._left_motor = self._robot.getDevice("left wheel motor")
        self._right_motor = self._robot.getDevice("right wheel motor")
        self._left_motor.setPosition(float('inf'))
        self._right_motor.setPosition(float('inf'))
        
        self._keyboard = self._robot.getKeyboard()
        self._keyboard.enable(self._timestep)

        if self._lidar is None:
            raise ValueError("Lidar device not found. Ensure the lidar device is correctly named and exists in the Webots world file.")

    
    def world2map(self, gps_position, world_size, floor_size):
        # gps_position: tuple (x, y, z) where x and y are in meters
        # world_size: tuple (width, height) of the world in meters
        # floor_size: size of each grid cell in meters

        # Calculate grid/map position
        x, y, z = gps_position
        map_x = int((x + world_size[0] / 2) // floor_size)
        map_y = int((y + world_size[1] / 2) // floor_size)

        return map_x, map_y
    
    
    @property
    def robot(self):
        return self._robot

    @property
    def timestep(self):
        return self._timestep

    @property
    def lidar(self):
        return self._lidar

    @property
    def left_motor(self):
        return self._left_motor

    @property
    def right_motor(self):
        return self._right_motor
        
    @property
    def keyboard(self):
        return self._keyboard
    
    @property
    def gps(self):
        return self._gps
        
    @property
    def compass(self):
        return self._compass
