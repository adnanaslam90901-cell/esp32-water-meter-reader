"""Convert binary/colored display masks from the Kaggle dataset to YOLO ROI boxes.

Expected layout below DATASET_DIR (names may be nested): image JPG files and mask
files whose stem contains the image stem. Review generated labels before training.
"""
from pathlib import Path
import argparse, shutil
import cv2
import numpy as np

def main(root: Path, out: Path):
    images=[p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg'} and 'mask' not in p.stem.lower()]
    masks=[p for p in root.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'} and 'mask' in str(p).lower()]
    (out/'images').mkdir(parents=True,exist_ok=True); (out/'labels').mkdir(parents=True,exist_ok=True)
    count=0
    for image_path in images:
        candidates=[m for m in masks if image_path.stem in m.stem or m.stem in image_path.stem]
        if not candidates: continue
        image=cv2.imread(str(image_path)); mask=cv2.imread(str(candidates[0]),cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None: continue
        if mask.shape!=image.shape[:2]: mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
        ys,xs=np.where(mask>20)
        if not len(xs): continue
        x1,x2,y1,y2=xs.min(),xs.max(),ys.min(),ys.max(); h,w=image.shape[:2]
        xc=((x1+x2)/2)/w; yc=((y1+y2)/2)/h; bw=(x2-x1+1)/w; bh=(y2-y1+1)/h
        target=f'{count:05d}'; shutil.copy2(image_path,out/'images'/f'{target}.jpg')
        (out/'labels'/f'{target}.txt').write_text(f'0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')
        count+=1
    (out/'dataset.yaml').write_text(f'path: {out.resolve()}\ntrain: images\nval: images\nnames:\n  0: reading_window\n')
    print(f'Prepared {count} image/mask pairs in {out}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--out',type=Path,default=Path('prepared_roi'));a=p.parse_args();main(a.dataset,a.out)
