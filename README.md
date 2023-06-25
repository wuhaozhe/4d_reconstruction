## MMFace4D: A Large-Scale Multi-Modal 4D Face Dataset for Audio-Driven 3D Face Animation
------
Haozhe Wu, Jia Jia, Junliang Xing, Hongwei Xu, Xiangyuan Wang, Jelo Wang
[[Paper]](https://arxiv.org/abs/2303.09797)


![plot](./images/demo.png)

This repo gives the official code of the paper MMFace4D. The source code of 4D reconstruction, mesh sequence compression, and face animation baseline is given.

### Environments
------

### 4D Reconstruction
------

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


### Audio-Driven Face Animation
------

### License and Citation
------

### Acknowledgements
------
