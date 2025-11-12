#!/bin/bash
# 启动TensorBoard查看训练日志

# 修改为你的模型目录
MODEL_DIR="d:/CodesFile/RedBlue/UAV_RL/results/uav_env/MADDPG_0_2"

# 如果在Linux上，使用相对路径
if [ ! -d "$MODEL_DIR" ]; then
    MODEL_DIR="../results/uav_env/MADDPG_0_2"
fi

echo "启动TensorBoard..."
echo "模型目录: $MODEL_DIR"
echo "访问地址: http://localhost:6006"
echo "按 Ctrl+C 停止"

tensorboard --logdir="$MODEL_DIR" --port=6006
