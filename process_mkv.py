'''
    preprocess and compress mkv to nut and mp4 files
    use vad to clip videos
'''

import torch
import torchaudio
import open3d as o3d
import os
import numpy as np
import re
import shutil
import ffmpeg
import cv2
import argparse
from tqdm import tqdm

USE_ONNX = False # change this to True if you want to test onnx model
SAMPLING_RATE = 16000  
model, utils = torch.hub.load(repo_or_dir='/home/wuhz/.cache/torch/hub/snakers4_silero-vad_master',
                              model='silero_vad',
                              source = 'local',
                              onnx=USE_ONNX)
                              
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

def get_frame_number(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

def compress_mkv(mkv_path_list, audio_path, folder, video_fps = 30):
    wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE, min_speech_duration_ms = 500)
    if len(speech_timestamps) == 0:
        return 0

    frame_number = get_frame_number(mkv_path_list[0])
    start_time = speech_timestamps[0]['start']
    end_time = speech_timestamps[-1]['end']
    start_frame = int(start_time * video_fps / SAMPLING_RATE)
    end_frame = int(end_time * video_fps / SAMPLING_RATE) + 2
    start_frame = max(start_frame, 0)
    end_frame = min(end_frame, frame_number)
    # video frame选取的范围是左闭右开[start_frame, end_frame)
    start_audio_index = int(max((start_frame / 30) * SAMPLING_RATE, 0))
    end_audio_index = int(min((end_frame / 30) * SAMPLING_RATE, wav.shape[0]))
    # audio选择的范围同样是左闭右开[start_audio_index, end_audio_index)

    wav_clip = wav[start_audio_index: end_audio_index]
    dst_audio_path = os.path.join(folder, audio_path.split('/')[-1])
    torchaudio.save(dst_audio_path, wav_clip.unsqueeze(0), SAMPLING_RATE)

    for idx, mkv_path in enumerate(mkv_path_list):
        file_name = mkv_path.split('/')[-1].split('.')[-2]
        color_video_path = os.path.join(folder, file_name + "_{}.mp4".format(idx))
        depth_video_path = os.path.join(folder, file_name + "_{}.nut".format(idx))
        intrinsic_path = os.path.join(folder, file_name + "_{}.json".format(idx))
        
        reader = o3d.io.AzureKinectMKVReader()
        reader.open(mkv_path_list[idx])
        metadata = reader.get_metadata()
        o3d.io.write_azure_kinect_mkv_metadata(intrinsic_path, metadata)

        process_rgb = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', framerate=video_fps, s='{}x{}'.format(1920, 1080))
            .output(color_video_path, pix_fmt='yuv420p', vcodec='libx264', r=video_fps, loglevel="quiet", crf=10)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        process_depth = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='gray16le', framerate=video_fps, s='{}x{}'.format(1920, 1080))
            .output(depth_video_path, pix_fmt='gray16le', vcodec='ffv1', r=video_fps, loglevel="quiet")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        frame_counter = 0
        counter = 0
        while not reader.is_eof():
            rgbd = reader.next_frame()
            if rgbd is None:
                continue
            np_color = np.asarray(rgbd.color).copy()[:, :]
            np_depth = np.asarray(rgbd.depth).copy()
            if start_frame <= frame_counter and frame_counter < end_frame:
                process_rgb.stdin.write(np_color.tobytes())
                process_depth.stdin.write(np_depth.tobytes())
                counter += 1
            frame_counter += 1

        reader.close()
        process_rgb.stdin.close()
        process_rgb.wait()
        process_depth.stdin.close()
        process_depth.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mkv_list', nargs='+', required = True)
    parser.add_argument('--audio_path', type=str, required = True)
    parser.add_argument('--folder', type=str, required = True)

    args = parser.parse_args()
    compress_mkv(args.mkv_list, args.audio_path, args.folder)