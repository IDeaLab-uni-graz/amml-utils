from nc_py_api import Nextcloud
import os
import zipfile


class MissingEnvironmentVariable(Exception):
    pass


def nextcloud_login():
    # create Nextcloud client instance class
    print("Setting up Nextcloud connection...")
    if ("NEXTCLOUD_URL" not in os.environ
       or "NEXTCLOUD_USERNAME" not in os.environ
       or "NEXTCLOUD_PASSWORD" not in os.environ):
        raise MissingEnvironmentVariable("Required environment variables for nextcloud connection do not exist!")
    nc = Nextcloud(
        nextcloud_url=os.getenv("NEXTCLOUD_URL"),
        nc_auth_user=os.getenv("NEXTCLOUD_USERNAME"),
        nc_auth_pass=os.getenv("NEXTCLOUD_PASSWORD"))
    return nc


def download_from_nextcloud(dataset_name, data_path):
    tmp_path = os.path.join(data_path, "tmp_files.zip")

    if "DATASETS_DIRECTORY" not in os.environ:
        raise MissingEnvironmentVariable("Required environment variable 'DATASETS_DIRECTORY' (determining the "
                                         "directory of the datasets in nextcloud) does not exists!")
    nc_dataset_directory = os.path.join(os.getenv("DATASETS_DIRECTORY"), dataset_name)

    nc = nextcloud_login()

    print(f"Downloading files for dataset '{dataset_name}'...")
    zip_path = nc.files.download_directory_as_zip(nc_dataset_directory, tmp_path)

    print(f"Extracting files for dataset '{dataset_name}'...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    os.remove(tmp_path)
