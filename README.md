## MMFace4D: A Large-Scale Multi-Modal 4D Face Dataset for Audio-Driven 3D Face Animation
------
Haozhe Wu, Jia Jia, Junliang Xing, Hongwei Xu, Xiangyuan Wang, Jelo Wang
[[Paper]](https://arxiv.org/abs/2303.09797)


![plot](./images/demo.png)

This repo gives the official code of the paper MMFace4D. The source code of 4D reconstruction, mesh sequence compression, and face animation baseline is given.

### Environments
------

For offline render, we need to install OSMesa, please follow the instructions of [pyrender](https://pyrender.readthedocs.io/en/latest/install/index.html)

For the differential render, we leverage nvdiffrast, please follow the instructions of [nvdiffrast](https://github.com/NVlabs/nvdiffrast)

Afterwards, run the other environments with 
`pip install -r requirements.txt`

### 4D Reconstruction
------

The reconstruction code is implemented in the `reconstruction` folder. We respectively provide the code of reconstructing 3D faces from three RGBD cameras and one RGBD camera.

For three-camera reconstruction, run

`python multicam_reconstruction.py --file_path ../test_data/000337 --save_path ../test_data/000337_save --faces_path ../test_data/faces.pkl`

For one-camera reconstruction, run

`
python singlecam_reconstruction.py --file_path ../test_data/000337 --save_path ../test_data/000337_save --faces_path ../test_data/faces.pkl
`

### Mesh Sequence Compression
------
Our mesh compression algorithm compress 3D mesh sequence with the same topology. Here we give an example of a compresses vertices to video files.

**Encode**
```python
from mesh_compression.encode import SequenceEncoder
encoder = SequenceEncoder('test_data/test', number_of_vertex)   # we have three video files, test_data/test_{0, 1, 2}.nut
for i in range(len(vertex_sequences)): # vertex_sequences has shape of frame_num * num_vertex * 3
    frame = vertex_sequences[i]
    encoder.write_frame(frame)

encoder.close()
```

**Decode**
```python
from mesh_compression.decode import decode_nut_to_np
vertices = decode_nut_to_np('./test_data/test', number_of_vertex)
```

### Preprocess Azure Kinect RGBD files
------
With the recorded MKV files of azure kinect, we decode it to nut files, mp4 files, and wav files. The nut files records depth video, mp4 files record RGB video, wav files record audio.
The preprocess code is provided in `process_mkv.py`.


### License and Citation
------
Cite
```
@article{wu2023mmface4d,
  title={MMFace4D: A Large-Scale Multi-Modal 4D Face Dataset for Audio-Driven 3D Face Animation},
  author={Wu, Haozhe and Jia, Jia and Xing, Junliang and Xu, Hongwei and Wang, Xiangyuan and Wang, Jelo},
  journal={arXiv preprint arXiv:2303.09797},
  year={2023}
}
```

### Acknowledgements
------
