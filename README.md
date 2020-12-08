# Yolov4-deepsort-head-detection

Head detector trained using [darknet](https://github.com/AlexeyAB/darknet) Yolov4 model. Tracker trained using [cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning) model.

## Environment
- `Ubuntu 18.04.5 LTS`
- `NVIDIA GeForce RTX 2080 Ti`
- `CUDA 10.0 / CuDNN 7.6.3`
- `Python 3.7.9`
- `Tensorflow 2.3.0rc0 GPU`

---

## Used dataset
- Custom dataset \+ about 100 images

---

## Dependencies
- python
    - opencv-python, numpy
    - tensorflow
    - matplotlib
- CUDA 10.0 / CuDNN 7.6.3
- darknet
    - **For Yolov4 Object Detection**
    - `libdarknet.so`
        - Compiled with `GPU=1`, `CUDNN=1`, `OPENCV=1`, `LIBSO=1`


## prepare environment and run
```python
pip install -r requirements.txt
```

```python
python detectandtrack.py
```
***you should modify code to make change***



