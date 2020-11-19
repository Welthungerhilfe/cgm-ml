import zipfile
import numpy as np
import utils

ENCODING = 'charmap'
utils.setWidth(int(240 * 0.75))
utils.setHeight(int(180 * 0.75))
width = utils.getWidth()
height = utils.getHeight()

def process(calibration_fname: str, pcd_fpath: str):
    # Convert to depthmap
    calibration = utils.parseCalibration(calibration_fname)
    points = utils.parsePCD(pcd_fpath)
    output = np.zeros((width, height, 3))
    for p in points:
        v = utils.convert3Dto2D(calibration[1], p[0], p[1], p[2])
        x = int(width - v[0] - 1)
        y = int(height - v[1] - 1)
        if x >= 0 and y >= 0 and x < width and y < height:
            output[x][y][0] = p[3]
            output[x][y][2] = p[2]
    return output
def write_depthmap(output_depth_fpath: str, depthmap):
    # Write depthmap
    with open('data', 'wb') as f:
        header_str = str(width) + 'x' + str(height) + '_0.001_255\n'
        f.write(header_str.encode(ENCODING))
        for y in range(height):
            for x in range(width):
                depth = int(depthmap[x][y][2] * 1000)
                confidence = int(depthmap[x][y][0] * 255)
                depth_byte = chr(int(depth / 256)).encode(ENCODING)
                depth_byte2 = chr(depth % 256).encode(ENCODING)
                confidence_byte = chr(confidence).encode(ENCODING)
                f.write(depth_byte)
                f.write(depth_byte2)
                f.write(confidence_byte)
    # Zip data
    with zipfile.ZipFile(output_depth_fpath, "w", zipfile.ZIP_DEFLATED) as f:
        f.write('data', 'data')
        f.close()
    # Visualsiation for debug
    #print str(width) + "x" + str(height)
    #plt.imshow(output)
    #plt.show()
