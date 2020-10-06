import math
import matplotlib.pyplot as plt
from PIL import Image
from pyntcloud import PyntCloud


def selectData(artifacts, model_id):
    df = artifacts[artifacts['model_id'] == model_id]
    df = df.sort_values(by=['error'])
    df = df.reset_index()
    return df


def imageshow(df, result_grid_filename, result_figsize_resolution):
    grid_size = math.ceil(math.sqrt(len(df['rgb_file'])))
    fig, axes = plt.subplots(grid_size,
                             grid_size,
                             figsize=(result_figsize_resolution,
                                      result_figsize_resolution))
    current_file_number = 0
    for image_filename in df['rgb_file']:
        x_position = current_file_number % grid_size
        y_position = current_file_number // grid_size
        image = Image.open(image_filename)
        rotated = image.transpose(Image.ROTATE_270)
        axes[x_position, y_position].imshow(rotated)

        current_file_number += 1

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    plt.savefig(result_grid_filename)


def pcdrgb(df, index):
    image = Image.open(df.at[index, 'rgb_file'])
    image = image.resize((300, 160), Image.ANTIALIAS)
    image = image.transpose(Image.ROTATE_270)
    output = PyntCloud.from_file(df.at[index, 'pcd_file'])
    output.plot()
