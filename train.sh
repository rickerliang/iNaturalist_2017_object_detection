export PYTHONPATH=$PYTHONPATH:/home/lyk/machine_learning/Supervised_Learning/tensorflow_model/research/:/home/lyk/machine_learning/Supervised_Learning/tensorflow_model/research/slim
CUDA_VISIBLE_DEVICES=0 python object_detection/train.py --logtostderr --pipeline_config_path=faster_rcnn_resnet101_coco.config --train_dir=train_dir
