import argparse
import os
import shutil
import subprocess
import tempfile
from ultralytics import YOLO
from PIL import Image


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_annotated(result, out_path):
    try:
        img_arr = result.plot()
        img = Image.fromarray(img_arr)
        img.save(out_path)
    except Exception:
        # fallback handled by caller
        raise


def run_generator(generator_cmd, dataroot, name):
    # generator_cmd should be a string with placeholders {dataroot} and {name}
    cmd = generator_cmd.format(dataroot=dataroot, name=name)
    print('Running generator command:')
    print(cmd)
    # run through shell so complex commands can be used
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f'Generator command failed with exit {res.returncode}')


def detect_and_label(weights, image_dir, out_base, conf, device):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f'Generated image directory not found: {image_dir}')

    labels_dir = os.path.join(out_base, 'labels')
    ann_dir = os.path.join(out_base, 'annotated_images')
    ensure_dir(labels_dir)
    ensure_dir(ann_dir)

    print('Loading YOLO model:', weights)
    model = YOLO(weights)

    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    if not files:
        print('No images found in generated dir:', image_dir)
        return 0,0

    processed = 0
    fires = 0
    for fname in files:
        img_path = os.path.join(image_dir, fname)
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, stem + '.txt')
        ann_path = os.path.join(ann_dir, fname)

        try:
            results = model.predict(source=img_path, imgsz=640, conf=conf, verbose=False, device=device)
        except Exception as e:
            print(f'Error running model on {img_path}:', e)
            continue

        r = results[0]
        try:
            num_boxes = len(r.boxes)
        except Exception:
            try:
                num_boxes = len(r.boxes.data)
            except Exception:
                num_boxes = 0

        label = '1' if num_boxes > 0 else '0'
        if label == '1':
            fires += 1
        processed += 1

        with open(label_path, 'w') as fh:
            fh.write(label)

        try:
            save_annotated(r, ann_path)
        except Exception:
            try:
                img = Image.open(img_path).convert('RGB')
                img.save(ann_path)
            except Exception as e:
                print('Failed to save annotated or copy original for', img_path, e)

        if processed % 100 == 0:
            print(f'Processed {processed} images, fires so far: {fires}')

    return processed, fires


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='Folder with source RGB images to convert (if running generator)', default=None)
    parser.add_argument('--generator_cmd', help='Optional generator command template with {dataroot} and {name} placeholders', default=None)
    parser.add_argument('--name', help='Experiment name used by generator (for output path templates)', default='forest_test')
    parser.add_argument('--gen_results_dir', help='If generator already ran, path to generated images directory; skips generation', default=None)
    parser.add_argument('--gen_output_template', help='Template for generator output images directory, e.g. pccoe/results/{name}/test_latest/images', default='pccoe/results/{name}/test_latest/images')
    parser.add_argument('--weights', default='yolov8n.pt', help='YOLO weights')
    parser.add_argument('--out_dir', default='thermal_detection_results', help='Output base dir for labels and annotated images')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection')
    parser.add_argument('--device', default=None, help='Device for inference (e.g. 0 or cpu)')
    args = parser.parse_args()

    # Decide generated images directory
    gen_dir = None
    tmpdir = None
    try:
        if args.gen_results_dir:
            gen_dir = args.gen_results_dir
            print('Using provided generated images dir:', gen_dir)
        elif args.generator_cmd and args.input_dir:
            # prepare temporary dataroot and copy images
            tmpdir = os.path.abspath(os.path.join('tmp_gen_runs', args.name))
            if os.path.exists(tmpdir):
                print('Cleaning previous tmp dir', tmpdir)
                shutil.rmtree(tmpdir)
            os.makedirs(tmpdir, exist_ok=True)
            # copy input images into dataroot
            print('Copying input images to dataroot:', tmpdir)
            for f in os.listdir(args.input_dir):
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    shutil.copy(os.path.join(args.input_dir,f), tmpdir)
            # run generator
            run_generator(args.generator_cmd, tmpdir, args.name)
            gen_dir = args.gen_output_template.format(name=args.name)
            print('Assuming generator wrote images to:', gen_dir)
        else:
            # fallback: try default template
            candidate = args.gen_output_template.format(name=args.name)
            if os.path.isdir(candidate):
                gen_dir = candidate
                print('Found generated images dir:', gen_dir)
            else:
                raise ValueError('No generated images dir found and no generator command provided. Provide --gen_results_dir or both --generator_cmd and --input_dir')

        processed, fires = detect_and_label(args.weights, gen_dir, args.out_dir, args.conf, args.device)
        print(f'Finished. Processed {processed} images, detected fire in {fires}. Outputs in: {args.out_dir}')
    finally:
        # do not automatically remove tmpdir; keep for inspection
        pass

if __name__ == '__main__':
    main()
