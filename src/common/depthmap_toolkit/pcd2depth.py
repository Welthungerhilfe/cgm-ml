import zipfile

import numpy as np

from utils import parse_pcd, convert_3d_to_2d

ENCODING = 'charmap'


def process(calibration, pcd_fpath: str, width: int, height: int):
    # Convert to depthmap
    points = parse_pcd(pcd_fpath)
    output = np.zeros((width, height, 3))
    for p in points:
        v = convert_3d_to_2d(calibration[1], p[0], p[1], p[2], width, height)
        x = int(width - v[0] - 1)
        y = int(height - v[1] - 1)
        if x >= 0 and y >= 0 and x < width and y < height:
            output[x][y][0] = p[3]
            output[x][y][2] = p[2]
    return output


def write_depthmap(output_depth_fpath: str, depthmap, width: int, height: int):
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
