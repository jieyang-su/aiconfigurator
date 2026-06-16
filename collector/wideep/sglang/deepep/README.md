Guidance for collecting deepep data in normal and low-latency modes.

Notes:
- MASTER_ADDR: IP address of the node with RANK=0.
- WORLD_SIZE: total number of nodes.
- RANK: 0-based index for this node.
- {num_node}: total number of nodes (e.g., 2 or 4).
- xxx: GPU type/model (e.g., A100, H100).

# Build Docker

Note: The test files under `collector/wideep/sglang/deepep/` are sourced from [DeepEP](https://github.com/deepseek-ai/DeepEP/tree/main/tests) with some modifications applied.

```bash
docker build -t deepep:latest -f docker/Dockerfile.deepep .
docker run -it --network host --gpus all -v aiconfigurator/collector/wideep/sglang/deepep:/new_workspace --privileged deepep:latest bash
```

# Two-node configuration

Server:
```bash
export MASTER_ADDR=10.6.131.20
export WORLD_SIZE=2
export MASTER_PORT=40303
export RANK=0
```

Client:
```bash
export MASTER_ADDR=10.6.131.20
export WORLD_SIZE=2
export MASTER_PORT=40303
export RANK=1
```

# Four-node configuration

Server:
```bash
export MASTER_ADDR=10.6.131.13
export WORLD_SIZE=4
export MASTER_PORT=40303
export RANK=0
```

Client:
```bash
export MASTER_ADDR=10.6.131.13
export WORLD_SIZE=4
export MASTER_PORT=40303
export RANK=1

export MASTER_ADDR=10.6.131.13
export WORLD_SIZE=4
export MASTER_PORT=40303
export RANK=2

export MASTER_ADDR=10.6.131.13
export WORLD_SIZE=4
export MASTER_PORT=40303
export RANK=3
```

# Test intra-node mode

Run the following command on a single node:
```bash
python /new_workspace/test_intranode.py \
  --num-processes 8 \
  --num-tokens 4096 \
  --hidden 7168 \
  --num-topk 8 \
  --num-experts 256 \
  --num-sms 20 \
  |& tee deepep_node_1_mode_normal_tok4096.log
```

Repeat for the token points required by AIC, for example
`128,512,2048,4096`.

# Test inter-node normal mode

On the first node:
Note: Replace {num_node} with the total number of nodes (e.g., 2 or 4).
```bash
python /new_workspace/test_internode.py |& tee deepep_node_{num_node}_mode_normal.log
```
On the other node(s):
```bash
python /new_workspace/test_internode.py
```

For node_num > 1, the DeepEP script must be launched concurrently on every
node. All nodes use the same `MASTER_ADDR`, `MASTER_PORT`, and `WORLD_SIZE`;
each node uses its own `RANK`.

Node 0:

```bash
export MASTER_ADDR=<node0_ip>
export MASTER_PORT=40303
export WORLD_SIZE=4
export RANK=0

python /new_workspace/test_internode.py \
  --num-processes 8 \
  --hidden 7168 \
  --num-topk 8 \
  --num-experts 256 \
  --tokens 128,512,2048,4096 \
  --num-sms 20 \
  |& tee deepep_node_4_mode_normal.log
```

Node 1/2/3 run the same command with `RANK=1`, `2`, or `3`. Rank 0 logs are
used by `extract_data.py` to write final perf txt files; non-zero ranks only
participate in the distributed DeepEP test.

# Test low-latency mode

On the first node:
Note: Replace {num_node} with the total number of nodes (e.g., 2 or 4).
```bash
python /new_workspace/test_internode.py  --test-ll-compatibility |& tee deepep_node_{num_node}_mode_ll.log
```
On the other node(s):
```bash
python /new_workspace/test_internode.py  --test-ll-compatibility
```

# Post-process log files
Save the processed deepep data under path path/to/aiconfigurator/src/aiconfigurator/systems/data/xxx/sglang/<sglang_version>/.
Replace xxx with the GPU type (e.g., A100). Point --log-dir to that directory.
```bash
python aiconfigurator/collector/wideep/sglang/deepep/extract_data.py --log-dir path/to/aiconfigurator/src/aiconfigurator/systems/data/xxx/sglang/<sglang_version>/
```

For H20 / DeepSeek-V3, prefer writing explicit metadata:

```bash
python aiconfigurator/collector/wideep/sglang/deepep/extract_data.py \
  --log-dir path/to/logs \
  --output-normal path/to/wideep_deepep_normal_perf.txt \
  --output-ll path/to/wideep_deepep_ll_perf.txt \
  --node-num 4 \
  --framework sglang \
  --version 0.0.0.dev1+g959d8a09d \
  --device "NVIDIA H20" \
  --kernel-source deepep
```
