export PYTHONPATH=$PYTHONPATH:/home/lyk/machine_learning/Supervised_Learning/tensorflow_model/research/:/home/lyk/machine_learning/Supervised_Learning/tensorflow_model/research/slim
CUDA_VISIBLE_DEVICES=3 python object_detection/eval.py --logtostderr --pipeline_config_path=faster_rcnn_resnet50_coco.config --checkpoint_dir=train_dir --eval_dir=eval_dir
