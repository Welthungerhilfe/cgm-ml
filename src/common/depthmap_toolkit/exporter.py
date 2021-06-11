import logging
import logging.config

import numpy as np
from depthmap import Depthmap

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


def export_obj(filename: str,
               dmap: Depthmap,
               triangulate: bool):
    """Export .obj file, which can be visualized in tools like Meshlab.

    triangulate=True generates OBJ of type mesh
    triangulate=False generates OBJ of type pointcloud
    """
    count = 0
    indices = np.zeros((dmap.width, dmap.height))

    # Create MTL file (a standart extension of OBJ files to define geometry materials and textures)
    material = filename[:len(filename) - 4] + '.mtl'
    if dmap.has_rgb:
        with open(material, 'w') as f:
            f.write('newmtl default\n')
            f.write('map_Kd ../' + dmap.rgb_fpath + '\n')

    with open(filename, 'w') as f:
        if dmap.has_rgb:
            f.write('mtllib ' + material[filename.index('/') + 1:] + '\n')
            f.write('usemtl default\n')
        for x in range(2, dmap.width - 2):
            for y in range(2, dmap.height - 2):
                depth = dmap.parse_depth(x, y)
                if not depth:
                    continue
                res = dmap.convert_2d_to_3d_oriented(1, x, y, depth)
                if not res:
                    continue
                count = count + 1
                indices[x][y] = count  # add index of written vertex into array
                f.write('v ' + str(res[0]) + ' ' + str(res[1]) + ' ' + str(res[2]) + '\n')
                f.write('vt ' + str(x / dmap.width) + ' ' + str(1 - y / dmap.height) + '\n')

        if triangulate:
            _do_triangulation(dmap, indices, f)
        logging.info('Mesh exported into %s', filename)


def _do_triangulation(dmap: Depthmap, indices, filehandle):
    max_diff = 0.2
    for x in range(2, dmap.width - 2):
        for y in range(2, dmap.height - 2):
            # get depth of all points of 2 potential triangles
            d00 = dmap.parse_depth(x, y)
            d10 = dmap.parse_depth(x + 1, y)
            d01 = dmap.parse_depth(x, y + 1)
            d11 = dmap.parse_depth(x + 1, y + 1)

            # check if first triangle points have existing indices
            if indices[x][y] > 0 and indices[x + 1][y] > 0 and indices[x][y + 1] > 0:
                # check if the triangle size is valid (to prevent generating triangle
                # connecting child and background)
                if abs(d00 - d10) + abs(d00 - d01) + abs(d10 - d01) < max_diff:
                    c = str(int(indices[x][y]))
                    b = str(int(indices[x + 1][y]))
                    a = str(int(indices[x][y + 1]))
                    # define triangle indices in (world coordinates / texture coordinates)
                    filehandle.write('f ' + a + '/' + a + ' ' + b + '/' + b + ' ' + c + '/' + c + '\n')

            # check if second triangle points have existing indices
            if indices[x + 1][y + 1] > 0 and indices[x + 1][y] > 0 and indices[x][y + 1] > 0:
                # check if the triangle size is valid (to prevent generating triangle
                # connecting child and background)
                if abs(d11 - d10) + abs(d11 - d01) + abs(d10 - d01) < max_diff:
                    a = str(int(indices[x + 1][y + 1]))
                    b = str(int(indices[x + 1][y]))
                    c = str(int(indices[x][y + 1]))
                    # define triangle indices in (world coordinates / texture coordinates)
                    filehandle.write('f ' + a + '/' + a + ' ' + b + '/' + b + ' ' + c + '/' + c + '\n')


def _write_pcd_header(filehandle, count):
    filehandle.write('# timestamp 1 1 float 0\n')
    filehandle.write('# .PCD v.7 - Point Cloud Data file format\n')
    filehandle.write('VERSION .7\n')
    filehandle.write('FIELDS x y z c\n')
    filehandle.write('SIZE 4 4 4 4\n')
    filehandle.write('TYPE F F F F\n')
    filehandle.write('COUNT 1 1 1 1\n')
    filehandle.write('WIDTH ' + count + '\n')
    filehandle.write('HEIGHT 1\n')
    filehandle.write('VIEWPOINT 0 0 0 1 0 0 0\n')
    filehandle.write('POINTS ' + count + '\n')
    filehandle.write('DATA ascii\n')


def export_pcd(filename: str, dmap: Depthmap):
    with open(filename, 'w') as f:
        count = str(_get_count(dmap))
        _write_pcd_header(f, count)

        for x in range(2, dmap.width - 2):
            for y in range(2, dmap.height - 2):
                depth = dmap.parse_depth(x, y)
                if not depth:
                    continue
                res = dmap.convert_2d_to_3d(1, x, y, depth)
                if not res:
                    continue
                confidence = dmap.parse_confidence(x, y)
                f.write(str(-res[0]) + ' ' + str(res[1]) + ' ' + str(res[2]) + ' ' + str(confidence) + '\n')
        logging.info('Pointcloud exported into %s', filename)


def _get_count(dmap: Depthmap) -> int:
    count = 0
    for x in range(2, dmap.width - 2):
        for y in range(2, dmap.height - 2):
            depth = dmap.parse_depth(x, y)
            if not depth:
                continue
            res = dmap.convert_2d_to_3d(1, x, y, depth)
            if not res:
                continue
            count = count + 1
    return count
