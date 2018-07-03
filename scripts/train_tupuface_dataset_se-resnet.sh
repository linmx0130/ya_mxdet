/usr/local/python-3.6.5/bin/python3 ./train_faster_rcnn.py \
    --epochs 20 \
    --gpus="2" \
    --batch_size 1 \
    --learning_rate 0.0005 \
    --save_path="/world/data-gpu-112/zhanglinghan/face-detect-faster-rcnn-mx/faster-rcnn-se_resnet50_v2-9anchors" \
    --save_interval 10000 \
    --model="se_resnet50_v2" \
    --feature_name="se_resnetv20_stage4_activation0_output" \
    --pretrained_model="/world/data-gpu-112/zhanglinghan/face-detect-faster-rcnn-mx/faster-rcnn-se_resnet50_v2-9anchors/faster-rcnn-se_resnet50_v2-9anchors-150000.gluonmodel"
