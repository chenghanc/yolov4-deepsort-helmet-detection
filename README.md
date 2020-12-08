# Yolov4-Deepsort-Head-Detection

Head detector trained using [darknet](https://github.com/AlexeyAB/darknet) Yolov4 model. Tracker trained using [cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning) model.

## Environment
- `Ubuntu 18.04.5 LTS`
- `NVIDIA GeForce RTX 2080 Ti`
- `CUDA 10.0 / CuDNN 7.6.3`
- `Python 3.7.9`
- `Tensorflow 2.3.0rc0 GPU`

---

## Used dataset
- Custom head dataset \+ about 10000 images

## Dependencies
- python
    - opencv-python, numpy, scipy
    - tensorflow
    - matplotlib
- CUDA 10.0 / CuDNN 7.6.3
- darknet
    - **For Yolov4 Object Detection**
    - `libdarknet.so`
        - Compiled with `GPU=1`, `CUDNN=1`, `OPENCV=1`, `LIBSO=1`


## Run the code

```python
python headtracking.py
```

---

# Appendix

## How to perform tracking head with deepsort

- First train a **Yolov4 Detector** on custom head dataset
- Do another training for cosine metric learning
    - Get cropped head dataset
    - Write a dataset adapter similar to the existing ones
    - Train a **Tracker**
    


