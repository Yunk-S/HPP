# DeepSpeed ZeRO-3 Hang/Deadlock 问题诊断和解决方案
# 
# 问题症状：
# - 训练在第3个迭代左右卡死
# - NCCL 超时错误
# - 不同 rank 之间的日志输出不同步
#
# 诊断步骤：
# 1. 设置 HPP_HANG_DEBUG=1 查看每个 rank 的执行流程
# 2. 设置 NCCL_DEBUG=INFO 查看 NCCL 通信详情
# 3. 使用 py-spy dump 查看卡住时的 Python 栈

# ===============================================
# 推荐的训练启动脚本 (用于 4 卡 DeepSpeed ZeRO-3)
# ===============================================

# 诊断环境变量 (生产环境可注释)
export HPP_HANG_DEBUG=1
export HPP_HANG_DEBUG_SYNC=1  # 使用 CUDA 同步精确定位 (会很慢，仅用于诊断)

# NCCL 诊断 (生产环境可注释)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# DeepSpeed 优化参数
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0

# 内存优化
export CUDA_MODULE_LOADING=LAZY

# 启动命令
# 使用 DeepSpeed launcher (推荐)
deepspeed --num_gpus=4 train.py --config=hpp

# 或使用 torchrun (需先设置 DeepSpeed 插件)
# torchrun --nproc_per_node=4 --master_port=29500 train.py --config=hpp
