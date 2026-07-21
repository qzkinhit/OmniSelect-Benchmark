# 5 个新增数据集的池/验证/测试划分

补齐 `experiments/split_manifests/`(旧 4 数据集协议)未覆盖的 5 个数据集:ETTh2、
DaISy CSTR、DaISy steamgen、CIFAR-10、CIFAR-100N。每个文件含 `pool_ids`、`val_ids`、
`test_ids` 与随机种子配方(`rng_recipe`),是该次运行里**全部方法共享**的候选池/
验证/测试划分,不是某个具体方法选中的子集(每个方法在候选池内部具体选了哪些样本,
目前只有 `results.json` 里的 `sel_sha12` 单向哈希,尚不能反查,见
`docs/REPRODUCIBILITY.md` 的说明)。复现命令见 README.md 的"结果"一节。
