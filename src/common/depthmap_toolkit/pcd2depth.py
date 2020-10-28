import zipfile

import numpy as np

import utils


def process(calibration, pcd, depthfile):

    #read PCD and calibration
    calibration = utils.parseCalibration(calibration)
    points = utils.parsePCD(pcd)
    utils.setWidth(int(240 * 0.75))
    utils.setHeight(int(180 * 0.75))

    #convert to depthmap
    width = utils.getWidth()
    height = utils.getHeight()
    output = np.zeros((width, height, 3))
    for p in points:
        v = utils.convert3Dto2D(calibration[1], p[0], p[1], p[2])
        x = int(width - v[0] - 1)
        y = int(height - v[1] - 1)
        if x >= 0 and y >= 0 and x < width and y < height:
            output[x][y][0] = p[3]
            output[x][y][2] = p[2]

    #write depthmap
    with open('data', 'wb') as file:
        file.write(str(width) + 'x' + str(height) + '_0.001_255\n')
        for y in range(height):
            for x in range(width):
                depth = int(output[x][y][2] * 1000)
                confidence = int(output[x][y][0] * 255)
                file.write(chr(depth / 256))
                file.write(chr(depth % 256))
                file.write(chr(confidence))

    #zip data
    with zipfile.ZipFile(depthfile, "w", zipfile.ZIP_DEFLATED) as zip:
        zip.write('data', 'data')
        zip.close()

    #visualsiation for debug
    #print str(width) + "x" + str(height)
    #plt.imshow(output)
    #plt.show()
