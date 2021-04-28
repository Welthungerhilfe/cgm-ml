#parse line of numbers
def parseNumbers(line):
    output = []
    values = line.split(" ")
    for value in values:
        output.append(float(value))
    return output

#parse calibration file
def parseCalibration(filepath):
    global calibration
    with open(filepath, 'r') as file:
        calibration = []
        file.readline()[:-1]
        calibration.append(parseNumbers(file.readline()))
        #print(str(calibration[0]) + '\n') #color camera intrinsics - fx, fy, cx, cy
        file.readline()[:-1]
        calibration.append(parseNumbers(file.readline()))
        #print(str(calibration[1]) + '\n') #depth camera intrinsics - fx, fy, cx, cy
        file.readline()[:-1]
        calibration.append(parseNumbers(file.readline()))
        #print(str(calibration[2]) + '\n') #depth camera position relativelly to color camera in meters
        calibration[2][1] *= 8.0  #workaround for wrong calibration data
    return calibration

#convert point into 3D
def convert_2d_to_3d(intrisics, x, y, z):
  fx = intrisics[0] * float(width)
  fy = intrisics[1] * float(height)
  cx = intrisics[2] * float(width)
  cy = intrisics[3] * float(height)
  tx = (x - cx) * z / fx
  ty = (y - cy) * z / fy
  output = []
  output.append(tx)
  output.append(ty)
  output.append(z)
  return output

#convert point into 2D
def convert3Dto2D(intrisics, x, y, z):
  fx = intrisics[0] * float(width)
  fy = intrisics[1] * float(height)
  cx = intrisics[2] * float(width)
  cy = intrisics[3] * float(height)
  tx = x * fx / z + cx
  ty = y * fy / z + cy
  output = []
  output.append(tx)
  output.append(ty)
  output.append(z)
  return output



#getter
def getWidth():
  return width

#getter
def getHeight():
  return height

#setter
def setWidth(value):
  global width
  width = value

#setter
def setHeight(value):
  global height
  height = value
