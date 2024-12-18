import gdown
import os
import sys


def download_from_googledrive(fname: str, id: str):

    if os.path.exists(fname):

        print(f"{sys._getframe(0).f_code.co_name} - File exist, pass.")

    else:

        print(f"{sys._getframe(0).f_code.co_name} - Download files.")

        dir: str = os.path.dirname(fname)
        os.makedirs(dir, exist_ok=True)
        gdown.download(id=id, output=fname, verify=False)