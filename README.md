# Deep3D
Real-Time end-to-end 2D-to-3D Video Conversion, based on deep learning.<br> Inspired by [piiswrong/deep3d](https://github.com/piiswrong/deep3d), we rebuild the network on pytorch and optimize it in time domain and faster inference speed. So, try it and enjoy your own 3D movies.

<div align="center">
  <img src="./medias/wood_result_360p.gif"><br>
</div>

Left is input video and right is output video with [parallax](https://en.wikipedia.org/wiki/Parallax).<br>

More examples:
- [wood-1080p](https://www.bilibili.com/video/bv1bt4y1x7N3)
- [Journey to the West (86)](https://www.bilibili.com/video/bv1xT4y1B7QK)

## Inference speed

|           Plan           | 360p (FPS) | 720p (FPS) | 1080p (FPS) | 4k (FPS) |
| :----------------------: | :--------: | :--------: | :---------: | :------: |
|       GPU (2080ti)       |     84     |     87     |     77      |    26    |
| CPU (Xeon Platinum 8260) |    27.7    |    14.1    |     7.2     |   2.0    |

## Run Deep3D
### Prerequisites
- Linux, Mac OS, Windows
- Python 3.7+
- [ffmpeg 3.4.6+](http://ffmpeg.org/)
- [Pytorch 1.7.1](https://pytorch.org/)
- CPU or NVIDIA GPU<br>

### Dependencies
This code depends on opencv-python available via pip install.
```bash
pip install opencv-python
```

### Clone this repo
```bash
git clone https://github.com/HypoX64/Deep3D
cd Deep3D
```

### Get Pre-Trained Models
You can download pre_trained models from:
[[Google Drive]](https://drive.google.com/drive/folders/1o-JRU9A38rHwoozHZNJDxKKAydgK_z04?usp=sharing) [[百度云,提取码xxo0 ]](https://pan.baidu.com/s/1Qml48TBI7_AC_d5oiZEAyQ) <br>
Note:
- 360p can get the best result.
- The published models are not inference optimized.
- Models are still under training, 1080p and 4k models will be uploaded in the future.


### Run it!
```bash
python inference.py --model ./export/deep3d_v1.0_640x360_cuda.pt --video ./medias/wood.mp4 --out ./result/wood.mp4 --inv
# some video need to reverse left and right views (--inv)
```


## Acknowledgements
This code borrows heavily from [[deep3d]](https://github.com/piiswrong/deep3d) [[DeepMosaics]](https://github.com/HypoX64/DeepMosaics)