import os, sys, re, datetime, argparse
import numpy as np
import cv2, torch

from steerable_pyramid import SteerablePyramid, SuboctaveSP
from phase_based_processing import PhaseBased
from phase_utils import *

EPS = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ap = argparse.ArgumentParser()

ap.add_argument("-v", "--video_path", type=str, required=True)
ap.add_argument("-a", "--phase_mag", type=float, default=25.0)
ap.add_argument("-lo", "--freq_lo", type=float, required=True)
ap.add_argument("-hi", "--freq_hi", type=float, required=True)
ap.add_argument("-n", "--colorspace", type=str, default="luma3",
                choices={"luma1", "luma3", "gray", "yiq", "rgb"})
ap.add_argument("-p", "--pyramid_type", type=str, default="half_octave",
                choices={"full_octave","half_octave",
                         "smooth_half_octave","smooth_quarter_octave"})
ap.add_argument("-s", "--sigma", type=float, default=0.0)
ap.add_argument("-t", "--attenuate", type=bool, default=False)
ap.add_argument("-fs","--sample_frequency", type=float, default=-1.0)
ap.add_argument("-r", "--reference_index", type=int, default=0)
ap.add_argument("-c", "--scale_factor", type=float, default=1.0)
ap.add_argument("-b", "--batch_size", type=int, default=2)
ap.add_argument("-d", "--save_directory", type=str, default="")

if __name__ == '__main__':

    args = vars(ap.parse_args())

    video_path = args["video_path"]
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        sys.exit()

    save_dir = args["save_directory"] or os.path.dirname(video_path)
    if not os.path.exists(save_dir):
        print(f"Save directory not found, using input video directory")
        save_dir = os.path.dirname(video_path)

    video_name = re.search(r"\w+(?=\.\w+)", video_path).group()
    video_save = os.path.join(save_dir, f"{video_name}_{args['colorspace']}_{int(args['phase_mag'])}x.mp4")

    # Colorspace setup
    if args["colorspace"] == "luma1":
        colorspace_func = lambda x: bgr2yiq(x)[:, :, 0]
        inv_colorspace = lambda x: cv2.cvtColor(
            cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
            cv2.COLOR_GRAY2BGR)

    elif args["colorspace"] in {"luma3","yiq"}:
        colorspace_func = bgr2yiq
        inv_colorspace = lambda x: cv2.cvtColor(
            cv2.normalize(yiq2rgb(x), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3),
            cv2.COLOR_RGB2BGR)

    elif args["colorspace"] == "gray":
        colorspace_func = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        inv_colorspace = lambda x: cv2.cvtColor(
            cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
            cv2.COLOR_GRAY2BGR)

    else:  # rgb
        colorspace_func = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        inv_colorspace = lambda x: cv2.cvtColor(
            cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3),
            cv2.COLOR_RGB2BGR)

    frames, video_fs = get_video(video_path, args["scale_factor"], colorspace_func)
    ref = frames[args["reference_index"]]
    h, w = ref.shape[:2]
    num_frames = len(frames)

    fs = args["sample_frequency"] if args["sample_frequency"] > 0 else video_fs
    transfer = bandpass_filter(args["freq_lo"], args["freq_hi"], fs, num_frames, DEVICE)

    max_depth = int(np.floor(np.log2(min(h,w))) - 2)
    pyr_type = args["pyramid_type"]

    if pyr_type == "full_octave":
        csp = SteerablePyramid(max_depth,4,1,1.0,True)
    elif pyr_type == "half_octave":
        csp = SteerablePyramid(max_depth,8,2,0.75,True)
    elif pyr_type == "smooth_half_octave":
        csp = SuboctaveSP(max_depth,8,2,6,True)
    else:
        csp = SuboctaveSP(max_depth,8,4,6,True)

    filters,_ = csp.get_filters(h,w,cropped=False)
    filters = torch.tensor(np.array(filters),dtype=torch.float32).to(DEVICE)

    batch = args["batch_size"]
    if filters.shape[0] % batch != 0:
        for b in range(batch,0,-1):
            if filters.shape[0] % b == 0:
                batch = b
                break

    frames_t = torch.tensor(np.array(frames),dtype=torch.float32).to(DEVICE)
    pb = PhaseBased(args["sigma"], transfer, args["phase_mag"], args["attenuate"],
                    args["reference_index"], batch, DEVICE, EPS)

    if args["colorspace"] in {"yiq","rgb"}:
        output = torch.zeros_like(frames_t)
        for c in range(frames_t.shape[-1]):
            dft = get_fft2_batch(frames_t[:,:,:,c]).to(DEVICE)
            output[:,:,:,c] = pb.process_single_channel(frames_t[:,:,:,c],filters,dft)

    elif args["colorspace"] == "luma3":
        output = frames_t.clone()
        dft = get_fft2_batch(frames_t[:,:,:,0]).to(DEVICE)
        output[:,:,:,0] = pb.process_single_channel(frames_t[:,:,:,0],filters,dft)

    else:
        dft = get_fft2_batch(frames_t).to(DEVICE)
        output = pb.process_single_channel(frames_t,filters,dft)

    output = output.cpu().numpy()
    out = cv2.VideoWriter(video_save, cv2.VideoWriter_fourcc(*'MP4V'),
                           int(np.round(video_fs)),
                           (int(w/args["scale_factor"]), int(h/args["scale_factor"])))

    for i in range(num_frames):
        frame = inv_colorspace(output[i])
        frame = cv2.resize(frame,(int(w/args["scale_factor"]),int(h/args["scale_factor"])))
        out.write(frame)

    out.release()
    print(f"Saved: {video_save}")