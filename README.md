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

## Example
- Based on model trained for Person Re-Identification

![](test-output.gif)

- Based on model trained for Head dataset

![](test-output-head.gif)

---

# Appendix

## How to perform head tracking with Deepsort

Deepsort is trained for Person Re-Identification. If you want to perform head tracking with Deepsort then it would need to re-train new model on head dataset.

- First train a **Yolov4 Head Detector** on custom head dataset using [darknet](https://github.com/AlexeyAB/darknet)
- Do another training for cosine metric learning using [cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning)
    - Get cropped head dataset (sequential, ideally)
        - Implement python code to crop head images based on **Yolov4 Head Detector** result
        - Or crop head images using darknet based on **Yolov4 Head Detector** result
    - Write a dataset adapter similar to the existing ones
    - Train / fine-tune a new **Tracker** and then combine with **Yolov4 Head Detector** to perform head tracking

- Training command (If training goes wrong with **cosine-softmax** loss, then try **triplet** loss)

```python
python train_veri.py --dataset_dir=./VeRi --loss_mode=triplet --log_dir=./output/head/ --run_id=triplet
```

- For reference [reference](https://github.com/nwojke/cosine_metric_learning/issues/46)


