import datetime
import os


def get_dataset(workspace, dataset_name, dataset_path):
    print("Accessing dataset...")
    if not os.path.exists(dataset_path):
        dataset = workspace.datasets[dataset_name]

        print("Downloading dataset.. Current date and time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        dataset.download(target_path=dataset_path, overwrite=False)

        print("Finished downloading, Current date and time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def get_dataset_path(data_dir, dataset_name):
    return str(data_dir / dataset_name)
