import os
import shutil
import random
from pathlib import Path
import typer

def get_corresponding_xml_file(txt_file: Path, xml_folder: Path) -> Path:
    folder_path = Path(os.path.join(xml_folder, txt_file))
    temp_list = list(folder_path.glob("*.xml"))
    if len(temp_list) != 1:
        raise ValueError(f"Found an xml folder with {len(temp_list)} elements for {txt_file}")
    file_path_xml = temp_list[0]
    return file_path_xml

def split_dataset(
    data_folder: Path = typer.Option(
        ...,
        help="Path to the data folder",
    ),
    dev_percentage: float = typer.Option(
        0.2,
        help="Percentage of the data to use for development",
    ),
    test_percentage: float = typer.Option(
        0.2,
        help="Percentage of the data to use for testing",
    ),
    if_dirs_exist: str = typer.Option(
        "fail",
        help="What to do if the train, dev or test folders already exist. Possible values: 'fail', 'overwrite'",
    ),
) -> None:


    random.seed(42)
    if (data_folder/"train").exists() or (data_folder/"dev").exists() or (data_folder/"test").exists():
        if if_dirs_exist=="fail":
            raise ValueError("The train, dev or test folders already exist.")
        if if_dirs_exist=="overwrite":
            print("The train, dev or test folders already exist. They will be overwritten.")
            shutil.rmtree(os.path.join(data_folder, 'train'), ignore_errors=True)
            shutil.rmtree(os.path.join(data_folder, 'dev'), ignore_errors=True)
            shutil.rmtree(os.path.join(data_folder, 'test'), ignore_errors=True)
    # Create the txt folders
    os.makedirs(os.path.join(data_folder, 'train/txt'), exist_ok=True)
    os.makedirs(os.path.join(data_folder, 'dev/txt'), exist_ok=True)
    os.makedirs(os.path.join(data_folder, 'test/txt'), exist_ok=True)
    # Create the xml folders
    os.makedirs(os.path.join(data_folder, 'train/xml'), exist_ok=True)
    os.makedirs(os.path.join(data_folder, 'dev/xml'), exist_ok=True)
    os.makedirs(os.path.join(data_folder, 'test/xml'), exist_ok=True)

    full_txt_path = os.path.join(data_folder, 'full/txt')
    full_xml_path = os.path.join(data_folder, 'full/xml')

    txt_files = [f for f in os.listdir(full_txt_path) if f.endswith('.txt')]
    random.shuffle(txt_files)



    dev_size = int(len(txt_files) * dev_percentage)
    test_size = int(len(txt_files) * test_percentage)
    train_size = len(txt_files) - dev_size - test_size

    print(f"Train size: {train_size}")
    print(f"Dev size: {dev_size}")
    print(f"Test size: {test_size}")
    for i, file in enumerate(txt_files):
        if i < train_size:
            xml_file_path = get_corresponding_xml_file(file, full_xml_path)
            xml_file_name = file.split(".txt")[0] + ".xml"
            output_txt_path = os.path.join(data_folder, 'train/txt', file)
            output_xml_path = os.path.join(data_folder, 'train/xml', xml_file_name)
            shutil.copyfile(os.path.join(full_txt_path, file), output_txt_path)
            shutil.copyfile(xml_file_path, output_xml_path)
        elif i < train_size + dev_size:
            xml_file_path = get_corresponding_xml_file(file, full_xml_path)
            xml_file_name = file.split(".txt")[0] + ".xml"
            output_txt_path = os.path.join(data_folder, 'dev/txt', file)
            output_xml_path = os.path.join(data_folder, 'dev/xml', xml_file_name)
            shutil.copyfile(os.path.join(full_txt_path, file), output_txt_path)
            shutil.copyfile(xml_file_path, output_xml_path)
        elif i < train_size + dev_size + test_size:
            xml_file_path = get_corresponding_xml_file(file, full_xml_path)
            xml_file_name = file.split(".txt")[0] + ".xml"
            output_txt_path = os.path.join(data_folder, 'test/txt', file)
            output_xml_path = os.path.join(data_folder, 'test/xml', xml_file_name)
            shutil.copyfile(os.path.join(full_txt_path, file), output_txt_path)
            shutil.copyfile(xml_file_path, output_xml_path)

    return None

if __name__ == "__main__":
    typer.run(split_dataset)
