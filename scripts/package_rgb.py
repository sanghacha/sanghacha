"""Create a minimal remote-inference packet containing RGB PNGs only."""
from pathlib import Path
import argparse
import zipfile

p=argparse.ArgumentParser()
p.add_argument('capture',type=Path)
args=p.parse_args()
files=sorted(args.capture.glob('frame_*/*_rgb.png'))
assert files,'Capture images not found'
with zipfile.ZipFile(args.capture/'rgb_inputs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for file in files:z.write(file,file.relative_to(args.capture).as_posix())
print('RGB images packaged:',len(files))
