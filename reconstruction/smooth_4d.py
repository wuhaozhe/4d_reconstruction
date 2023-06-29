import torch
import torch.nn.functional as F
import utils
import argparse
import sys
sys.path.append('..')
from mesh_compression.decode import decode_nut_to_np
from mesh_compression.encode import SequenceEncoder

def lowpass_filter(vertex_array):
    filter_half_len = 3

    with torch.no_grad():
        vertex_tensor = torch.from_numpy(vertex_array).cuda()
        vertex_len = vertex_tensor.shape[0]
        vertex_tensor = torch.permute(vertex_tensor, (1, 2, 0))
        
        vertex_tensor_pad = F.pad(vertex_tensor, (filter_half_len, filter_half_len), mode = 'replicate')
        vertex_tensor_pad = torch.permute(vertex_tensor_pad, (2, 0, 1))
        vertex_tensor_pad = vertex_tensor_pad.reshape(len(vertex_tensor_pad), -1).cpu().numpy()
        vertex_filt = utils.low_pass_filter(vertex_tensor_pad, 5, 30)[filter_half_len: -filter_half_len]
        vertex_filt = vertex_filt.reshape(vertex_len, -1, 3)

    return vertex_filt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required = True)
    parser.add_argument('--out_path', type=str, required = True)
    args = parser.parse_args()
    args.vertex_number = 35709
    vertex_array = decode_nut_to_np(args.file_path, args.vertex_number)
    filt_vertex_array = lowpass_filter(vertex_array)
    compression_encoder = SequenceEncoder(args.out_path, args.vertex_number)
    for i in range(len(filt_vertex_array)):
        compression_encoder.write_frame(filt_vertex_array[i])
    compression_encoder.close()