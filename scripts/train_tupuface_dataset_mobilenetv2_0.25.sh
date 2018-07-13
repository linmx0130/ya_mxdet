/usr/local/python-3.6.5/bin/python3 ./train_faster_rcnn.py \
    --epochs 20 \
    --gpus="1" \
    --batch_size 1 \
    --learning_rate 0.0005 \
    --save_path="/world/data-gpu-112/zhanglinghan/face-detect-faster-rcnn-mx/faster-rcnn-mobilenetv2_0.25-9anchors" \
    --save_interval 10000 \
    --model="mobilenetv2_0.25" \
    --feature_name="mobilenetv20_features_linearbottleneck12_batchnorm2_fwd_output" \
    --pretrained_model="/world/data-gpu-112/zhanglinghan/face-detect-faster-rcnn-mx/faster-rcnn-mobilenetv2_0.25-9anchors/faster-rcnn-mobilenetv2_0.25-9anchors-170000.gluonmodel"
