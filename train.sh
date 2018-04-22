export PYTHONPATH=$PYTHONPATH:/home/lyk/machine_learning/Supervised_Learning/tensorflow_model/research/:/home/lyk/machine_learning/Supervised_Learning/tensorflow_model/research/slim
python object_detection/train.py --num_clones=2 --ps_tasks=1 --logtostderr --pipeline_config_path=faster_rcnn_resnet50_coco.config --train_dir=train_dir
