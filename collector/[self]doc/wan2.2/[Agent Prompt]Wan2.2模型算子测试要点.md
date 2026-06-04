在aiconfigurator库中实现对于sglang框架下wan2.2（文图生视频）模型的适配，第一步要在collector中添加算子测算代码，遵循collector库的一般实现规范和通用方法逻辑。进行代码生成和测试：

1. 硬件环境：主要为hopper和blackwell系gpu（sm=90、100等）；尽量真实地还原sglang框架在这些硬件条件下各个子模块算子的执行行为
2. 输入参数：结合模型config并考虑框架中真实运行时可能的并行算法（TP、SP等）及其对算子维数等的影响，测试多种输入参数
3. 算子测试对象：对于collector库中已有实现的算子（如通用的gemm等）不用新建，对于已有实现补充维度即可；而对于比较特殊的wan模型特色算子或collector无实现的算子才实现新的collect代码。
4. 算子粒度：尽可能每个测试代码和生成数据表仅包含单个算子（且单卡级别执行），而非多个算子复合成的大模块（如已有的mla_module，是现有代码库的遗留问题）。

\*可能存在的问题：里面的USPAttention模块在SP时涉及多卡All2All或Ring通信，对于单卡级别测算可能存在问题。此处的原则和思路是，坚持算子测试执行的是单卡级别、细粒度的单个算子，对于算子和通信耦合的USPAttention模块等，要对其进行进一步拆解，剥离通信并找出其实际的单卡attn算子。SP通信等仅影响执行算子的形状。

备注：
1. ~~使用ljc01的conda环境~~
2. 建议使用Docker环境（本地环境总是出现大量环境依赖无法解决），本机存在0.5.10.post1的sglang，应该没有问题。参考命令：
```docker run --gpus '"device=7"' --ipc=host --rm -v /home/ai_lab/ljc/scale-up-sim/aiconfigurator:/workspace -w /workspace 10.110.181.132:5000/sglang:0.5.10.post1```
3. 目前本机GPU为8卡H100，前6张卡被占用，需要手动选择后两张卡。