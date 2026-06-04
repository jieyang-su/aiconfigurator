# Models 解析总结

**快速概括**：`models.py` 及其 `BaseModel` 定位于模型结构抽象层，负责将原始的模型结构参数（从 JSON 等读出）或预定义的拓扑（Architecture）解耦，并实例化为具体的算子组合与性能元数据对象。它是联系“顶层并发调度策略”与“底层算子级耗时统计”的**算力需求图纸**。

## 1. 主要功能概括

*   **预处理与算力结构参数下发**：将原始的 HuggingFace 或定制模型配置（如 `num_layers`, `hidden_size` 等）以及部署期的切分并行度（TP, PP, EP 等）融合成单一对象进行路由管理。
*   **KVCache 水位推算**：计算因不同注意力机制架构（如 GQA / MHA / MLA ）以及量化方式（如 INT8 / FP16）带来的各种差异化的 KV 内存开销大小。
*   **算子派发路由图**：构建并持有具体的网络算子图层列表（Context Ops 和 Generation Ops），向后端性能引擎提供逐个 Op 算子的维度尺寸描述，供其打入硬件特征数据库查表。

## 2. 细节内容深度解析

本组件的内部实现将繁杂的模型网络结构归纳成了高度一致的工厂与基类。

### 2.1 `BaseModel` 基类
所有具体模型的父基准类，用来维护模型运行时必要的硬件宏观属性、分片段属性与底层公共开销计算逻辑。
*   **`__init__` 初始化与切分布局**：接收输入的 `num_layers`, `num_heads`, `num_kv_heads` 等超参数，结合传入的 `ModelConfig` 并发度设置（如 `tp_size`, `pp_size`）进行层切、头切的数学对齐计算操作。如将总体 KV 头分配为单卡局部的 `_num_kv_heads_per_gpu`。
*   **`get_kvcache_elements_per_token()` (极度核心的模型架构分水岭)**：
    *   定义了在此模型下，单一 Token 需额外派生多少规模的 Cache Elements。
    *   **传统架构 (GQA/MHA)**：计算为常规逻辑算法 `num_kv_heads_per_gpu * head_size * num_layers * 2`。
    *   **新型 MLA 架构 (如 DeepSeek / Kimi)**：由于 Multi-head Latent Attention 的底层特性（Latent KV 被所有物理头共享且不再随注意力 TP 切分），直接绕过多头逻辑，计算算法变更为 `num_layers * (kv_lora_rank + qk_rope_head_dim)`，大幅重构了模型的仿真负担。
*   **`get_kvcache_bytes_per_sequence()`**：乘上具体的 `seq_len` 以及指定物理量化模式下的位宽系数（如 INT8 = 1 byte），最终得出实际推演占用的显存字节。

### 2.2 工厂化门面：`get_model` 与结构转换
函数 `get_model(model_path, model_config, backend_name)` 是生成实例的总调度器：
*   **信息提取与家族转换**：借由 `_get_model_info` 获取模型规格，再利用 `_architecture_to_model_family` 方法将五花八门的子模型或网规名（比如 `Qwen2ForCausalLM`）聚合映射为宏观家族代号（如 `LLAMA`）。
*   **量化精度下沉 (`_apply_model_quant_defaults`)**：根据不同的量化设定，重写各算盘的通信精度与位宽默认值。
*   **按家族子类派发与 Backend 耦合**：值得注意的是，模型实例化也会**逆向依赖**目前仿真的 `backend_name`！如果发现采用的是 DeepSeek 系列且使用的是 SGLang，并开启了 MoE/EP 分流策略，它会专门返回一个与框架底层通信组件适配的类（如 `WideEPDeepSeekModel` 或 `SGLangEPMOEModel`）来刻画 `deepep` 下的定制通信开销与算子组排。

## 3. aic 支持的模型家族大类 (`model_family`)
基于底层算则逻辑差异，`inference_session` 及仿真内核支持的模型被高度归纳为以下几大基准簇，特征如下：

1. **LLAMA 家族 (`LLAMA`)**
    *   包含子类：`LlamaForCausalLM`, `Qwen2/3ForCausalLM`, `MiMoForCausalLM` 等。
    *   **特征**：代表业界标准的稠密（Dense）网络。以传统的 GQA/MHA 作为注意力实现基调，无特殊的路由或降维拓扑操作。
2. **常规 MoE 家族 (`MOE`, `HYBRIDMOE`)**
    *   包含子类：`MixtralForCausalLM`, `Qwen2MoeForCausalLM`, `Llama4ForConditionalGeneration` 等。
    *   **特征**：具有特化的 FFN 路由以及 Expert 选择策略。`HYBRIDMOE` 则刻画了部分层稠密、部分层路混合的模型。
3. **DeepSeek 家族 (`DEEPSEEK`, `DEEPSEEKV32`, `DEEPSEEKV4`) & Kimi (`KIMIK25`)**
    *   包含子类：`DeepseekV3ForCausalLM`, `KimiK25ForConditionalGeneration` 等。
    *   **特征**：属于近一阶段先进**复杂模型的仿真重灾区**。它集成了 MLA（极大地缓解算力仿真时显存水位）以及细粒度共享专家等混合 MoE 设计；并且可以因特殊通信工具（如 TensorRT、DeepEP）启用专为广度专家并行的结构类。
4. **Nemotron / NAS 家族 (`NEMOTRONNAS`, `NEMOTRONH`)**
    *   包含子类：`NemotronForCausalLM`, `DeciLMForCausalLM`。
    *   **特征**：基于（Neural Architecture Search）神经架构搜索动态装配配置产生的非固定规格流水线 Block 混合堆叠网络。
5. **GPT 家族 (`GPT`)**
    *   **特征**：经典的老派因果层机制大面基座。

## 4. 上层调用生态全景
模型类处于架构极底的物理结构刻画地位，它如何影响全局：
1. **依附生命周期**：在 `InferenceSession` `__init__` 解析最初阶段被创建并永久持有为内部核心属性 `self.model`。
2. **生成性能账单**：具体的 `Backend`（无论 SGLang 还是 TRT-LLM），其之所以能跑通底层时延推演 `_run_static_breakdown()`，就是高频**调用 `self.model.context_ops` 与 `self.model.generation_ops` 抽取模型层所构建的巨长图纸表**，从而拿到该层的确切乘加维度后丢入库文件进行毫秒查表。
3. **主导显存防线**：Backend 的 `_get_memory_usage()` 方法通过反复向模型发出如 `get_kvcache_bytes_per_sequence` 的测算请求，去拦截及标记每一个推演步骤是否存在击破 OOM 边界的危机。