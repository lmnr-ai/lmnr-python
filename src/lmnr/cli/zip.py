import os
from pathlib import Path
import zipfile


def zip_directory(directory_path: Path, zip_file_path: Path):
    with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory_path):
            for file in files:
                # Don't include the zip file itself, otherwise goes to infinite loop
                if file == zip_file_path.name:
                    continue

                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory_path)
                zipf.write(file_path, arcname)
