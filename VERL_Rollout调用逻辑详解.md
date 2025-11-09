# VERL Rollout 调用逻辑详解

## 📋 目录
- [1. Rollout 概述](#1-rollout-概述)
- [2. 调用链路](#2-调用链路)
- [3. 同步模式 (Sync Mode)](#3-同步模式-sync-mode)
- [4. 异步模式 (Async Mode)](#4-异步模式-async-mode)
- [5. 混合引擎机制](#5-混合引擎机制)
- [6. 多轮对话与工具调用](#6-多轮对话与工具调用)
- [7. 具体代码示例](#7-具体代码示例)
- [8. 性能优化](#8-性能优化)

---

## 1. Rollout 概述

### 1.1 什么是 Rollout？

**Rollout** 是 PPO 训练中负责**序列生成**的组件。它使用当前策略模型（Actor）从给定的 prompt 生成 response。

### 1.2 Rollout 的作用

```
┌────────────┐
│  Prompts   │  "解决这道数学题：2+3=?"
└─────┬──────┘
      │
      ▼
┌────────────────────────────────┐
│      Rollout Engine            │
│  (vLLM/SGLang/HuggingFace)     │
└─────┬──────────────────────────┘
      │
      ▼
┌────────────┐
│ Responses  │  "解答：2+3=5"
└────────────┘
```

### 1.3 支持的 Rollout 引擎

| 引擎 | 优势 | 适用场景 | 模式支持 |
|------|------|---------|---------|
| **vLLM** | 高吞吐、PagedAttention | 推理优化 | sync, async |
| **SGLang** | RadixAttention、多轮优化 | 工具调用、多轮对话 | sync, async |
| **HuggingFace** | 简单易用、兼容性好 | 调试、小规模 | sync |
| **Naive** | 最基础实现 | 测试 | sync |

---

## 2. 调用链路

### 2.1 完整调用栈

```
RayPPOTrainer.fit()
    │
    ├─> actor_rollout_wg.generate_sequences(batch)
    │       │
    │       ├─> [Ray RPC 调用所有 workers]
    │       │
    │       └─> ActorRolloutRefWorker.generate_sequences(prompts)
    │               │
    │               ├─> [混合引擎] 切换到 rollout 模式
    │               │   await self.rollout_mode()
    │               │
    │               ├─> self.rollout.generate_sequences(prompts)
    │               │       │
    │               │       ├─> [vLLM] vLLMRollout.generate_sequences()
    │               │       ├─> [SGLang] SGLangRollout.generate_sequences()
    │               │       └─> [HF] HFRollout.generate_sequences()
    │               │
    │               └─> [混合引擎] 切换回训练模式
    │                   await self.trainer_mode()
    │
    └─> 返回 DataProto(responses, log_probs, ...)
```

### 2.2 数据流转

```python
# 输入: DataProto
{
    "batch": {
        "input_ids": torch.Tensor,      # (batch_size, prompt_length)
        "attention_mask": torch.Tensor, # (batch_size, prompt_length)
        "position_ids": torch.Tensor,   # (batch_size, prompt_length)
    },
    "non_tensor_batch": {
        "uid": np.array,                # 唯一标识符
        "data_source": np.array,        # 数据来源
    },
    "meta_info": {
        "eos_token_id": int,
        "pad_token_id": int,
        "do_sample": bool,
    }
}

# ↓ Rollout 生成 ↓

# 输出: DataProto
{
    "batch": {
        "prompts": torch.Tensor,            # (batch_size, prompt_length)
        "responses": torch.Tensor,          # (batch_size, response_length)
        "input_ids": torch.Tensor,          # (batch_size, total_length)
        "attention_mask": torch.Tensor,     # (batch_size, total_length)
        "position_ids": torch.Tensor,       # (batch_size, total_length)
        "response_mask": torch.Tensor,      # (batch_size, response_length)
        "rollout_log_probs": torch.Tensor,  # (batch_size, response_length) [可选]
    },
    "non_tensor_batch": {...},
    "meta_info": {
        "timing": {
            "generate_sequences": float,    # 生成耗时
            "generation_timing/max": float,
            "generation_timing/min": float,
        }
    }
}
```

---

## 3. 同步模式 (Sync Mode)

### 3.1 配置

```yaml
actor_rollout_ref:
  rollout:
    name: vllm              # 引擎类型
    mode: sync              # 同步模式
    tensor_model_parallel_size: 4
    temperature: 1.0
    top_p: 0.9
    n: 16                   # 每个 prompt 生成数
```

### 3.2 工作流程

```python
# 在 ActorRolloutRefWorker 中
def generate_sequences(self, prompts: DataProto):
    """同步生成序列"""
    
    # 步骤 1: 数据准备
    prompts = prompts.to(device)
    
    # 步骤 2: [混合引擎] 切换到 Rollout 模式
    if self._is_actor:
        loop = get_event_loop()
        loop.run_until_complete(self.rollout_mode())
        # 释放 Actor 参数，加载 Rollout 权重
    
    # 步骤 3: 调用 Rollout 引擎生成
    output = self.rollout.generate_sequences(prompts=prompts)
    
    # 步骤 4: [混合引擎] 切换回训练模式
    if self._is_actor:
        loop.run_until_complete(self.trainer_mode())
        # 释放 Rollout 权重，加载 Actor 参数
    
    # 步骤 5: 返回结果
    output = output.to("cpu")
    return output
```

### 3.3 vLLM 同步生成示例

```python
class vLLMRollout(BaseRollout):
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """vLLM 同步批量生成"""
        
        # 1. 提取输入
        idx = prompts.batch["input_ids"]           # (batch_size, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.shape[0]
        
        # 2. 准备 vLLM 输入
        vllm_inputs = []
        for i in range(batch_size):
            # 过滤 padding tokens
            valid_mask = attention_mask[i] == 1
            valid_ids = idx[i][valid_mask].tolist()
            
            vllm_inputs.append(
                TokensPrompt(prompt_token_ids=valid_ids)
            )
        
        # 3. 调用 vLLM 引擎
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
        
        # 4. 解析输出
        response = []
        rollout_log_probs = []
        
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                # 提取生成的 token IDs
                response_ids = output.outputs[sample_id].token_ids
                response.append(response_ids)
                
                # 提取 log probabilities (如果需要)
                if self.config.calculate_log_probs:
                    curr_log_prob = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(curr_log_prob)
        
        # 5. Padding 到固定长度
        response = pad_2d_list_to_length(
            response, 
            self.pad_token_id, 
            max_length=self.config.response_length
        ).to(idx.device)
        
        if self.config.calculate_log_probs:
            rollout_log_probs = pad_2d_list_to_length(
                rollout_log_probs, 
                -1, 
                max_length=self.config.response_length
            ).to(idx.device).float()
        
        # 6. 构建完整序列
        seq = torch.cat([idx, response], dim=-1)
        
        # 7. 构建 position_ids
        response_length = response.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        new_position_ids = torch.cat([
            position_ids, 
            position_ids[:, -1:] + delta_position_id
        ], dim=-1)
        
        # 8. 构建 attention_mask
        response_attention_mask = (response != self.pad_token_id).long()
        new_attention_mask = torch.cat([
            attention_mask, 
            response_attention_mask
        ], dim=-1)
        
        # 9. 返回 DataProto
        batch = {
            "prompts": idx,
            "responses": response,
            "input_ids": seq,
            "attention_mask": new_attention_mask,
            "position_ids": new_position_ids,
            "response_mask": response_attention_mask,
        }
        
        if self.config.calculate_log_probs:
            batch["rollout_log_probs"] = rollout_log_probs
        
        return DataProto(batch=batch)
```

---

## 4. 异步模式 (Async Mode)

### 4.1 配置

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    mode: async             # 异步模式
    tensor_model_parallel_size: 4
    
    # 异步参数
    agent:
      num_workers: 4        # 并发 worker 数
      max_concurrent_requests: 128  # 最大并发请求数
    
    # 多轮对话
    multi_turn:
      enable: true
      max_user_turns: 16
      max_assistant_turns: 16
```

### 4.2 AgentLoopManager 架构

```
┌──────────────────────────────────────────────────────┐
│              AgentLoopManager                        │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │      AsyncLLMServerManager                 │    │
│  │  (负载均衡 + Sticky Session)                │    │
│  │                                            │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐    │    │
│  │  │ Server1 │  │ Server2 │  │ Server3 │    │    │
│  │  │ (vLLM)  │  │ (vLLM)  │  │ (vLLM)  │    │    │
│  │  └─────────┘  └─────────┘  └─────────┘    │    │
│  └────────────────────────────────────────────┘    │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │      AgentLoop (并发执行)                   │    │
│  │                                            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐ │    │
│  │  │ Loop 1   │  │ Loop 2   │  │ Loop N   │ │    │
│  │  │ (Sample) │  │ (Sample) │  │ (Sample) │ │    │
│  │  └──────────┘  └──────────┘  └──────────┘ │    │
│  └────────────────────────────────────────────┘    │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │      Tool/Environment                      │    │
│  │  (Code Executor, Search, Calculator)       │    │
│  └────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

### 4.3 异步生成流程

```python
class AgentLoopManager:
    """管理异步 Rollout 生成"""
    
    def __init__(self, config, worker_group, rm_wg):
        self.config = config
        self.worker_group = worker_group
        
        # 1. 创建 LLM Servers (vLLM/SGLang)
        self.server_handles = self._create_servers()
        
        # 2. 创建 Server Manager (负载均衡)
        self.server_manager = AsyncLLMServerManager(
            config=config,
            server_handles=self.server_handles
        )
        
        # 3. 初始化 Tokenizer/Processor
        self.tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
        self.processor = hf_processor(config.actor_rollout_ref.model.path)
        
        # 4. 创建 AgentLoop 类
        self.agent_loop_cls = self._get_agent_loop_class()
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """异步并发生成"""
        
        # 1. 将 batch 转换为单个样本列表
        batch_size = len(prompts.batch["input_ids"])
        samples = []
        for i in range(batch_size):
            sample = {
                "input_ids": prompts.batch["input_ids"][i],
                "attention_mask": prompts.batch["attention_mask"][i],
                # ... 其他字段
            }
            samples.append(sample)
        
        # 2. 并发执行 AgentLoop
        loop = asyncio.get_event_loop()
        outputs = loop.run_until_complete(
            self._concurrent_generate(samples)
        )
        
        # 3. 合并结果
        return self._merge_outputs(outputs)
    
    async def _concurrent_generate(self, samples):
        """并发生成多个样本"""
        tasks = []
        for sample in samples:
            # 为每个样本创建一个 AgentLoop 实例
            agent_loop = self.agent_loop_cls(
                trainer_config=self.config,
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            # 创建异步任务
            task = agent_loop.run(sample)
            tasks.append(task)
        
        # 并发执行所有任务
        outputs = await asyncio.gather(*tasks)
        return outputs
```

### 4.4 单个 AgentLoop 执行流程

```python
class AgentLoopBase:
    """单个样本的 Agent 循环"""
    
    async def run(self, sample: dict) -> AgentLoopOutput:
        """执行多轮对话 + 工具调用"""
        
        # 初始化
        request_id = str(uuid.uuid4())
        messages = sample["messages"]  # 初始对话
        
        prompt_ids = []
        response_ids = []
        response_mask = []
        response_logprobs = []
        num_turns = 0
        
        # 多轮循环
        for turn in range(self.config.max_turns):
            num_turns += 1
            
            # 步骤 1: 构建 prompt
            prompt_text = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            current_prompt_ids = self.tokenizer.encode(prompt_text)
            
            # 步骤 2: LLM 生成
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=current_prompt_ids,
                sampling_params={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_new_tokens,
                }
            )
            
            # 步骤 3: 解析生成结果
            generated_ids = output.token_ids
            generated_text = self.tokenizer.decode(generated_ids)
            
            # 记录 LLM 生成的 tokens
            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))
            if output.log_probs:
                response_logprobs.extend(output.log_probs)
            
            # 添加到消息历史
            messages.append({
                "role": "assistant",
                "content": generated_text
            })
            
            # 步骤 4: 检测工具调用
            tool_calls = self._parse_tool_calls(generated_text)
            
            if not tool_calls:
                # 没有工具调用，结束循环
                break
            
            # 步骤 5: 执行工具调用
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                
                # 调用工具
                tool_result = await self._execute_tool(
                    tool_name, 
                    tool_args
                )
                
                # 将工具结果 tokenize
                tool_result_text = json.dumps(tool_result)
                tool_result_ids = self.tokenizer.encode(tool_result_text)
                
                # 记录工具返回的 tokens (mask=0)
                response_ids.extend(tool_result_ids)
                response_mask.extend([0] * len(tool_result_ids))
                if output.log_probs:
                    response_logprobs.extend([0.0] * len(tool_result_ids))
                
                # 添加到消息历史
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_result_text
                })
        
        # 步骤 6: 计算奖励 (可选)
        reward_score = None
        if self.config.compute_reward_in_rollout:
            reward_score = await self._compute_reward(
                prompt_ids, 
                response_ids
            )
        
        # 返回结果
        return AgentLoopOutput(
            prompt_ids=current_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=reward_score,
            num_turns=num_turns,
        )
```

### 4.5 负载均衡机制

```python
class AsyncLLMServerManager:
    """LLM Server 负载均衡器"""
    
    def __init__(self, server_handles, max_cache_size=10000):
        self.server_handles = server_handles
        
        # 最少请求负载均衡 (Min-Heap)
        self.weighted_servers = [
            [0, (hash(server), server)] 
            for server in server_handles
        ]
        heapq.heapify(self.weighted_servers)
        
        # LRU 缓存：request_id -> server
        # 用于 Sticky Session (同一 request_id 总是发到同一 server)
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)
    
    def _choose_server(self, request_id: str):
        """选择 Server"""
        
        # 如果之前访问过，返回同一 server (Sticky Session)
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]
        
        # 选择请求数最少的 server
        server = self.weighted_servers[0][1][1]
        
        # 更新请求计数
        self.weighted_servers[0][0] += 1
        heapq.heapreplace(self.weighted_servers, self.weighted_servers[0])
        
        # 缓存映射
        self.request_id_to_server[request_id] = server
        
        return server
    
    async def generate(self, request_id, prompt_ids, sampling_params):
        """生成序列"""
        server = self._choose_server(request_id)
        
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        
        return output
```

---

## 5. 混合引擎机制

### 5.1 什么是混合引擎？

**混合引擎 (Hybrid Engine)** 允许在同一组 GPU 上**共享**生成引擎 (vLLM/SGLang) 和训练引擎 (FSDP)，通过动态切换模式实现高效利用。

```
同一块 GPU:
┌────────────────────────────────────┐
│                                    │
│  时刻 1: Rollout 模式               │
│  ┌──────────────────────────┐     │
│  │  vLLM Engine (推理)       │     │
│  │  - 加载 Rollout 权重      │     │
│  │  - 使用 PagedAttention    │     │
│  └──────────────────────────┘     │
│                                    │
│           ↓ 切换 ↓                 │
│                                    │
│  时刻 2: Trainer 模式               │
│  ┌──────────────────────────┐     │
│  │  FSDP Engine (训练)       │     │
│  │  - 加载 Actor 参数        │     │
│  │  - 加载 Optimizer         │     │
│  └──────────────────────────┘     │
│                                    │
└────────────────────────────────────┘
```

### 5.2 模式切换流程

```python
class ActorRolloutRefWorker:
    """支持混合引擎的 Worker"""
    
    def __init__(self, config, role):
        # 判断是否启用混合引擎
        self.hybrid_engine = config.rollout.hybrid_engine
        
        if self.hybrid_engine:
            self._is_actor = True      # 同时具有 Actor 能力
            self._is_rollout = True    # 同时具有 Rollout 能力
            
            # 初始化 Actor (FSDP)
            self.actor_module = self._init_actor_model()
            self.actor_optimizer = self._init_optimizer()
            
            # 初始化 Rollout (vLLM/SGLang)
            self.rollout = self._init_rollout_engine()
        else:
            # 独立模式：只有一种能力
            if role == "actor":
                self._is_actor = True
                self._is_rollout = False
            else:
                self._is_actor = False
                self._is_rollout = True
    
    async def rollout_mode(self):
        """切换到 Rollout 模式"""
        
        # 1. 卸载 Actor 参数到 CPU
        offload_fsdp_model_to_cpu(self.actor_module)
        offload_fsdp_optimizer(self.actor_optimizer)
        
        # 2. 同步权重到 Rollout 引擎
        weights = self._get_actor_weights()
        await self.rollout.update_weights(weights)
        
        # 3. 恢复 Rollout KV Cache 到 GPU
        await self.rollout.resume(tags=["weights", "kv_cache"])
        
        # 4. 清理显存
        torch.cuda.empty_cache()
        
        log_gpu_memory_usage("After switch to rollout mode")
    
    async def trainer_mode(self):
        """切换到训练模式"""
        
        # 1. 释放 Rollout 资源
        await self.rollout.release()
        
        # 2. 加载 Actor 参数到 GPU
        load_fsdp_model_to_gpu(self.actor_module)
        load_fsdp_optimizer(self.actor_optimizer)
        
        # 3. 清理显存
        torch.cuda.empty_cache()
        
        log_gpu_memory_usage("After switch to trainer mode")
    
    def generate_sequences(self, prompts):
        """生成序列 (带模式切换)"""
        
        # 如果是混合引擎，需要先切换模式
        if self._is_actor:
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
        
        # 调用 Rollout 引擎
        output = self.rollout.generate_sequences(prompts)
        
        # 切换回训练模式
        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
        
        return output
    
    def update_actor(self, data):
        """更新 Actor (训练)"""
        
        # 此时已经在 trainer_mode，直接训练
        # 前向传播
        logits = self.actor_module(
            input_ids=data.batch["input_ids"],
            attention_mask=data.batch["attention_mask"],
        )
        
        # 计算损失
        loss = self._compute_ppo_loss(logits, data)
        
        # 反向传播
        self.actor_optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.actor_module.parameters(),
            max_norm=1.0
        )
        
        # 更新参数
        self.actor_optimizer.step()
        
        return DataProto(meta_info={"metrics": {"loss": loss.item()}})
```

### 5.3 内存优化

```python
# 混合引擎显存分配示例 (A100 80GB)

# Rollout 模式:
# - vLLM 模型权重: 14GB (7B FP16)
# - KV Cache: 40GB
# - 临时激活: 5GB
# 总计: ~59GB

# Trainer 模式:
# - FSDP 模型参数: 7GB (Sharded)
# - FSDP 梯度: 7GB
# - Optimizer States: 28GB (AdamW)
# - 激活值: 10GB
# 总计: ~52GB

# 通过卸载优化:
actor.fsdp_config.param_offload: true       # 参数卸载到 CPU
actor.fsdp_config.optimizer_offload: true   # Optimizer 卸载到 CPU

# Trainer 模式优化后:
# - FSDP 参数 (GPU): 2GB (部分)
# - FSDP 参数 (CPU): 5GB
# - Optimizer (CPU): 28GB
# - 激活值: 10GB
# 总计 (GPU): ~12GB
```

---

## 6. 多轮对话与工具调用

### 6.1 配置

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: true
      max_user_turns: 16
      max_assistant_turns: 16
      tool_config_path: recipe/retool/sandbox_fusion_tool_config.yaml
      format: hermes  # 或 openai
```

### 6.2 工具配置示例

```yaml
# sandbox_fusion_tool_config.yaml
tools:
  - tool_schema:
      type: function
      function:
        name: code_interpreter
        description: Execute Python code in a sandbox environment
        parameters:
          type: object
          properties:
            code:
              type: string
              description: The Python code to execute
          required:
            - code
    
    tool_implementation:
      class_path: verl.workers.rollout.tools.sandbox
      class_name: SandboxCodeInterpreter
      config:
        timeout: 30
        max_memory: 512  # MB
```

### 6.3 多轮对话示例

```python
# 输入 Prompt:
messages = [
    {
        "role": "user",
        "content": "计算斐波那契数列的第10项"
    }
]

# ===== Turn 1: LLM 生成 =====
# Assistant 输出:
{
    "role": "assistant",
    "content": "我将使用 Python 代码来计算",
    "tool_calls": [
        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "arguments": {
                    "code": "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n\nresult = fib(10)\nprint(result)"
                }
            }
        }
    ]
}

# 生成的 tokens:
response_ids = [123, 456, 789, ...]  # "我将使用 Python..."
response_mask = [1, 1, 1, ...]       # 全为 1 (LLM 生成)

# ===== Turn 2: 工具执行 =====
# 执行代码，获得结果
tool_result = {
    "stdout": "55\n",
    "stderr": "",
    "exit_code": 0
}

# 工具返回的 tokens:
tool_result_text = json.dumps(tool_result)
tool_ids = tokenizer.encode(tool_result_text)

response_ids.extend(tool_ids)
response_mask.extend([0] * len(tool_ids))  # 工具输出 mask=0

# ===== Turn 3: LLM 继续生成 =====
# 添加工具结果到消息历史
messages.append({
    "role": "tool",
    "name": "code_interpreter",
    "content": tool_result_text
})

# LLM 基于工具结果继续生成
# Assistant 输出:
{
    "role": "assistant",
    "content": "斐波那契数列的第10项是 55"
}

# 最终的 tokens:
final_response_ids = [
    123, 456, 789,      # Turn 1: LLM 生成
    234, 567, 890,      # Turn 2: 工具返回
    345, 678, 901,      # Turn 3: LLM 生成
]

final_response_mask = [
    1, 1, 1,            # Turn 1: mask=1
    0, 0, 0,            # Turn 2: mask=0
    1, 1, 1,            # Turn 3: mask=1
]
```

### 6.4 Response Mask 的作用

```python
# response_mask 用于区分哪些 tokens 是模型生成的 (计算损失)

# 计算 PPO 损失时:
policy_loss = compute_policy_loss(
    old_log_prob=old_log_probs,      # (batch_size, response_length)
    log_prob=log_probs,              # (batch_size, response_length)
    advantages=advantages,           # (batch_size, response_length)
    response_mask=response_mask,     # (batch_size, response_length)
)

def compute_policy_loss(old_log_prob, log_prob, advantages, response_mask, ...):
    # 只对 response_mask=1 的位置计算损失
    ratio = torch.exp(log_prob - old_log_prob)
    
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
    
    policy_loss = -torch.min(surr1, surr2)
    
    # 应用 mask: 只计算 LLM 生成的 tokens
    policy_loss = policy_loss * response_mask
    
    # 平均 (只除以有效 tokens 数量)
    policy_loss = policy_loss.sum() / response_mask.sum()
    
    return policy_loss
```

---

## 7. 具体代码示例

### 7.1 端到端示例：单个训练步骤

```python
# 在 RayPPOTrainer.fit() 中

for batch_dict in train_dataloader:
    # ========== 阶段 1: 准备数据 ==========
    batch = DataProto.from_single_dict(batch_dict)
    # batch.batch["input_ids"]: (64, 2048)
    # batch.batch["attention_mask"]: (64, 2048)
    
    # 添加唯一标识
    batch.non_tensor_batch["uid"] = np.array([
        str(uuid.uuid4()) for _ in range(64)
    ])
    
    # ========== 阶段 2: Rollout 生成 ==========
    gen_batch = batch.pop(
        batch_keys=["input_ids", "attention_mask", "position_ids"]
    )
    
    # 每个 prompt 生成 16 个 responses
    gen_batch = gen_batch.repeat(repeat_times=16, interleave=True)
    # 现在 batch_size = 64 * 16 = 1024
    
    # 调用 Rollout (异步模式)
    if self.async_rollout_mode:
        gen_output = self.async_rollout_manager.generate_sequences(gen_batch)
    else:
        gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)
    
    # gen_output.batch:
    # - "prompts": (1024, 2048)
    # - "responses": (1024, 2048)
    # - "input_ids": (1024, 4096)
    # - "response_mask": (1024, 2048)
    # - "rollout_log_probs": (1024, 2048)  [如果启用]
    
    # ========== 阶段 3: 计算奖励 ==========
    batch = batch.repeat(repeat_times=16, interleave=True)
    batch = batch.union(gen_output)
    
    # 计算奖励分数
    reward_tensor, reward_extra_info = compute_reward(batch, self.reward_fn)
    batch.batch["token_level_scores"] = reward_tensor
    # token_level_scores: (1024, 2048)
    
    # ========== 阶段 4: 计算对数概率 ==========
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    batch = batch.union(old_log_prob)
    # old_log_probs: (1024, 2048)
    
    if self.use_reference_policy:
        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)
        # ref_log_probs: (1024, 2048)
    
    # ========== 阶段 5: 计算价值 ==========
    if self.use_critic:
        values = self.critic_wg.compute_values(batch)
        batch = batch.union(values)
        # values: (1024, 2048)
    
    # ========== 阶段 6: 计算优势 ==========
    # 应用 KL 惩罚 (如果启用)
    if self.config.algorithm.use_kl_in_reward:
        kld = (batch.batch["old_log_probs"] - 
               batch.batch["ref_log_probs"])
        batch.batch["token_level_rewards"] = (
            batch.batch["token_level_scores"] - 0.01 * kld
        )
    else:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    
    # 计算优势函数
    batch = compute_advantage(
        batch,
        adv_estimator=AdvantageEstimator.GRPO,
        gamma=1.0,
        lam=0.95,
    )
    # advantages: (1024, 2048)
    # returns: (1024, 2048)
    
    # ========== 阶段 7: 更新模型 ==========
    # 更新 Critic
    if self.use_critic:
        critic_output = self.critic_wg.update_critic(batch)
    
    # 更新 Actor
    actor_output = self.actor_rollout_wg.update_actor(batch)
```

### 7.2 自定义 Rollout 引擎

```python
from verl.workers.rollout.base import BaseRollout
from verl import DataProto

class CustomRollout(BaseRollout):
    """自定义 Rollout 引擎"""
    
    def __init__(self, config, model_config, device_mesh):
        super().__init__(config, model_config, device_mesh)
        
        # 初始化自定义引擎
        self.engine = self._init_custom_engine()
    
    def _init_custom_engine(self):
        # 实现自定义初始化
        pass
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """实现生成逻辑"""
        
        # 1. 提取输入
        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        
        # 2. 调用引擎生成
        outputs = self.engine.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.response_length,
            temperature=self.config.temperature,
        )
        
        # 3. 构建输出
        responses = outputs["sequences"]
        
        # 4. 返回 DataProto
        batch = {
            "prompts": input_ids,
            "responses": responses,
            "input_ids": torch.cat([input_ids, responses], dim=-1),
            # ... 其他字段
        }
        
        return DataProto(batch=batch)
    
    async def update_weights(self, weights):
        """更新权重 (混合引擎需要)"""
        for name, param in weights:
            self.engine.load_weight(name, param)
    
    async def release(self):
        """释放资源 (混合引擎需要)"""
        self.engine.clear_kv_cache()
        torch.cuda.empty_cache()
    
    async def resume(self, tags):
        """恢复资源 (混合引擎需要)"""
        if "kv_cache" in tags:
            self.engine.restore_kv_cache()

# 注册自定义 Rollout
from verl.workers.rollout.base import _ROLLOUT_REGISTRY

_ROLLOUT_REGISTRY[("custom", "sync")] = CustomRollout
```

### 7.3 自定义 AgentLoop

```python
from verl.experimental.agent_loop.agent_loop import AgentLoopBase

class CustomAgentLoop(AgentLoopBase):
    """自定义 Agent 循环"""
    
    async def run(self, sample):
        """执行自定义逻辑"""
        
        # 1. 初始化
        request_id = str(uuid.uuid4())
        messages = sample["messages"]
        
        # 2. 多轮对话
        for turn in range(self.config.max_turns):
            # 生成
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=self._prepare_prompt(messages),
                sampling_params=self._get_sampling_params(),
            )
            
            # 解析结果
            generated_text = self.tokenizer.decode(output.token_ids)
            
            # 自定义逻辑：检测特殊标记
            if "<END>" in generated_text:
                break
            
            # 工具调用
            if self._has_tool_call(generated_text):
                tool_result = await self._execute_tool(generated_text)
                messages.append({
                    "role": "tool",
                    "content": tool_result
                })
            else:
                break
        
        # 3. 返回结果
        return self._build_output(messages)
```

---

## 8. 性能优化

### 8.1 vLLM 优化参数

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    
    # TP 并行
    tensor_model_parallel_size: 4
    
    # 显存利用率
    gpu_memory_utilization: 0.9
    
    # KV Cache 配置
    max_num_seqs: 256           # 最大并发序列数
    max_num_batched_tokens: 8192  # 最大 batch tokens
    
    # 性能优化
    enable_prefix_caching: true  # 前缀缓存
    disable_log_stats: false     # 启用统计日志
    
    # 量化
    quantization: null  # awq, gptq, fp8
```

### 8.2 异步模式优化

```yaml
actor_rollout_ref:
  rollout:
    mode: async
    
    agent:
      num_workers: 8               # 并发 worker 数
      max_concurrent_requests: 256  # 最大并发请求
      
    # Server 数量 (负载均衡)
    num_servers: 4
```

### 8.3 Batch Balancing

```python
# 在 RayPPOTrainer 中
def _balance_batch(self, batch, metrics):
    """平衡各 DP rank 的计算负载"""
    
    # 1. 计算每个样本的有效 token 数
    attention_mask = batch.batch["attention_mask"]
    seqlen_lst = attention_mask.sum(-1).tolist()  # [batch_size]
    
    # 2. 将样本分配到各 DP rank (平衡负载)
    world_size = self.actor_rollout_wg.world_size
    partitions = get_seqlen_balanced_partitions(
        seqlen_lst,
        k_partitions=world_size,
        equal_size=True,
    )
    
    # 3. 重排序 batch
    global_idx = torch.tensor([
        j for partition in partitions for j in partition
    ])
    batch.reorder(global_idx)
    
    # 4. 记录负载不平衡度
    unbalance_stats = log_seqlen_unbalance(
        seqlen_list=seqlen_lst,
        partitions=partitions,
    )
    metrics.update(unbalance_stats)

# 效果:
# 未优化: Rank 0: 50000 tokens, Rank 1: 30000 tokens (不平衡)
# 优化后: Rank 0: 40000 tokens, Rank 1: 40000 tokens (平衡)
```

---

## 总结

### Rollout 关键要点

1. **职责**: 使用当前策略生成序列
2. **模式**: 
   - **Sync**: 批量同步生成，简单直接
   - **Async**: 并发异步生成，支持多轮对话和工具调用
3. **引擎**: vLLM、SGLang、HuggingFace
4. **混合引擎**: 生成和训练共享 GPU，动态切换模式
5. **优化**: TP 并行、前缀缓存、负载均衡、异步并发

### 使用建议

| 场景 | 推荐配置 |
|------|---------|
| **简单文本生成** | vLLM + Sync |
| **多轮对话** | SGLang + Async |
| **工具调用** | SGLang + Async + Multi-turn |
| **大模型 (>32B)** | vLLM + TP=8 |
| **显存受限** | Hybrid Engine + Offload |

---

**文档版本**: v1.0  
**最后更新**: 2025-01-06

