# SDAR MoE-Offloading 单样本实验分析

## 样本与总体结果

- 样本索引: 0
- 生成 token 数: 106
- 预测结果: `18`
- 参考答案: `18`
- 正确性: `True`
- 模型构建时间: 44.823 s
- 单样本生成总延迟: 48.763 s
- prefill 总延迟: 4251.985 ms
- decode block 数: 5
- decode step 总数: 55 (denoise=50, finalize=5)
- 平均每个 decode step 总延迟: 804.900 ms
- 平均每个 denoise step 总延迟: 795.912 ms
- 平均每个 finalize step 总延迟: 894.779 ms

## 专家命中统计

- layer-step 总数: 2640
- 请求的去重专家总数: 145128
- GPU cache 直接命中数: 36204 (24.95%)
- prefetch 命中数: 68779
- GPU cache + prefetch 总命中率: 72.34%
- 仍需 CPU miss load 的专家数: 40145 (27.66%)
- 平均每个 layer-step 请求专家数: 54.973
- 平均每个 layer-step 的 GPU cache 直接命中专家数: 13.714
- 平均每个 layer-step 的 prefetch 命中专家数: 26.053
- 平均每个 layer-step 的 CPU miss load 专家数: 15.206
- 平均每个 layer-step cache 内专家数: 16.000
- 平均每个 layer-step prefetch buffer 内可用专家数: 27.054

### CPU miss 最严重的 layer-step

| block | abs block | step | kind | layer | requested | cache hit | prefetch hit | cpu miss | miss rate |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 0 | denoise | 0 | 26 | 3 | 0 | 23 | 88.46% |
| 3 | 5 | 0 | denoise | 0 | 24 | 3 | 0 | 21 | 87.50% |
| 0 | 2 | 0 | denoise | 0 | 88 | 13 | 0 | 75 | 85.23% |
| 3 | 5 | 13 | finalize | 0 | 96 | 16 | 0 | 80 | 83.33% |
| 3 | 5 | 11 | denoise | 0 | 95 | 16 | 0 | 79 | 83.16% |
| 3 | 5 | 12 | denoise | 0 | 95 | 16 | 0 | 79 | 83.16% |
| 0 | 2 | 1 | finalize | 0 | 94 | 16 | 0 | 78 | 82.98% |
| 1 | 3 | 18 | denoise | 0 | 93 | 16 | 0 | 77 | 82.80% |
| 2 | 4 | 16 | finalize | 0 | 93 | 16 | 0 | 77 | 82.80% |
| 3 | 5 | 10 | denoise | 0 | 93 | 16 | 0 | 77 | 82.80% |

### CPU miss 最严重的层

| layer | requested | cache hit | prefetch hit | cpu miss | direct hit | combined hit | cpu miss rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3947 | 792 | 0 | 3155 | 20.07% | 20.07% | 79.93% |
| 1 | 3754 | 761 | 1533 | 1460 | 20.27% | 61.11% | 38.89% |
| 3 | 3240 | 751 | 1338 | 1151 | 23.18% | 64.48% | 35.52% |
| 5 | 3333 | 746 | 1514 | 1073 | 22.38% | 67.81% | 32.19% |
| 2 | 3107 | 803 | 1332 | 972 | 25.84% | 68.72% | 31.28% |
| 11 | 3344 | 737 | 1573 | 1034 | 22.04% | 69.08% | 30.92% |
| 47 | 3475 | 774 | 1627 | 1074 | 22.27% | 69.09% | 30.91% |
| 21 | 2925 | 773 | 1261 | 891 | 26.43% | 69.54% | 30.46% |
| 22 | 3135 | 793 | 1397 | 945 | 25.30% | 69.86% | 30.14% |
| 10 | 3519 | 766 | 1720 | 1033 | 21.77% | 70.65% | 29.35% |

## 时间轴

下面的时间轴来自详细 trace 中记录的 CUDA event。对每个 decode step，以该 step 起点记为 0 ms，再把所有 step 的同类操作起止点取平均。
这里的“平均开始/平均结束”是把一个 step 内 48 层里所有同类区间合并后，取最早开始和最晚结束得到的覆盖窗口；因此它描述的是该类操作在整步时间轴上的分布范围，不是某一个单独 op 的独占连续时长。
同理，“平均累计时长”是把该类操作在 48 层上的同类区间时长累加后再取平均；由于跨层重复和跨 stream 重叠都存在，各类操作的占比不会加总为 100%。

### 所有 decode step 的平均时间轴

| 操作 | 平均开始 | 平均结束 | 平均墙钟跨度 | 平均累计时长 | 平均占 step 比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention | 0.763 ms | 786.503 ms | 785.740 ms | 38.417 ms | 5.01% |
| routing | 2.346 ms | 787.039 ms | 784.693 ms | 17.154 ms | 2.22% |
| current-layer availability check | 3.122 ms | 787.520 ms | 784.398 ms | 3.757 ms | 0.48% |
| current-layer miss load | 3.238 ms | 799.540 ms | 796.303 ms | 398.301 ms | 50.53% |
| next-layer prefetch | 24.453 ms | 792.342 ms | 767.888 ms | 484.577 ms | 61.17% |
| reorder | 26.109 ms | 799.883 ms | 773.774 ms | 9.553 ms | 1.24% |
| gather | 26.334 ms | 802.401 ms | 776.067 ms | 133.019 ms | 14.46% |
| expert compute | 29.177 ms | 804.164 ms | 774.987 ms | 74.503 ms | 9.49% |
| scatter | 31.177 ms | 804.230 ms | 773.053 ms | 2.971 ms | 0.39% |
| cache promotion | 24.244 ms | 799.604 ms | 775.360 ms | 2.451 ms | 0.32% |

### 仅 denoise step 的平均时间轴

| 操作 | 平均开始 | 平均结束 | 平均墙钟跨度 | 平均累计时长 | 平均占 step 比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention | 0.766 ms | 777.900 ms | 777.134 ms | 39.307 ms | 5.16% |
| routing | 2.375 ms | 778.440 ms | 776.065 ms | 17.102 ms | 2.24% |
| current-layer availability check | 3.155 ms | 778.921 ms | 775.766 ms | 3.768 ms | 0.49% |
| current-layer miss load | 3.271 ms | 790.569 ms | 787.298 ms | 386.861 ms | 49.87% |
| next-layer prefetch | 24.060 ms | 783.491 ms | 759.430 ms | 475.888 ms | 60.95% |
| reorder | 25.699 ms | 790.912 ms | 765.213 ms | 9.601 ms | 1.26% |
| gather | 25.926 ms | 793.416 ms | 767.490 ms | 135.600 ms | 14.71% |
| expert compute | 28.760 ms | 795.155 ms | 766.395 ms | 73.710 ms | 9.51% |
| scatter | 30.731 ms | 795.221 ms | 764.490 ms | 2.967 ms | 0.39% |
| cache promotion | 23.852 ms | 790.632 ms | 766.780 ms | 2.431 ms | 0.32% |

### 仅 finalize step 的平均时间轴

| 操作 | 平均开始 | 平均结束 | 平均墙钟跨度 | 平均累计时长 | 平均占 step 比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention | 0.734 ms | 872.534 ms | 871.800 ms | 29.512 ms | 3.43% |
| routing | 2.053 ms | 873.027 ms | 870.973 ms | 17.671 ms | 2.00% |
| current-layer availability check | 2.796 ms | 873.505 ms | 870.710 ms | 3.650 ms | 0.41% |
| current-layer miss load | 2.904 ms | 889.258 ms | 886.353 ms | 512.701 ms | 57.15% |
| next-layer prefetch | 28.384 ms | 880.853 ms | 852.468 ms | 571.465 ms | 63.33% |
| reorder | 30.208 ms | 889.590 ms | 859.382 ms | 9.076 ms | 1.03% |
| gather | 30.417 ms | 892.252 ms | 861.835 ms | 107.205 ms | 11.96% |
| expert compute | 33.350 ms | 894.257 ms | 860.906 ms | 82.429 ms | 9.19% |
| scatter | 35.643 ms | 894.324 ms | 858.682 ms | 3.011 ms | 0.34% |
| cache promotion | 28.160 ms | 889.321 ms | 861.161 ms | 2.647 ms | 0.30% |

## nsys 交叉校验

- nsys capture 总时长: 48760.938 ms
- nsys prefill 总时长: 4251.874 ms
- nsys 统计到的 decode step 数: 55
- nsys 平均每个 decode step 总时长: 804.333 ms

说明: nsys 这里统计的是 NVTX push/pop 的 CPU 墙钟区间；trace 时间轴统计的是同名 NVTX 范围对应的 CUDA event 时间。对于 `next-layer prefetch` 这种异步 prefetch，CUDA event 更适合用于分析真实串并行关系。

## 串并行关系解读

1. `attention -> routing -> current-layer availability check -> current-layer miss load` 在同一层的主计算链上基本是严格串行的。当前层不先知道有哪些专家缺失，就无法发起 miss load。
2. `cache promotion` 很短，紧跟在 miss load 之后，本质是把本层刚加载的专家登记进 GPU cache。
3. `next-layer prefetch` 在当前层 miss load/cache promotion 之后启动，但它跑在单独的 prefetch stream 上，所以会和当前层后半段的 `reorder -> gather -> expert compute -> scatter` 产生明显重叠。
4. `reorder -> gather -> expert compute -> scatter` 是当前层真正消费专家权重做 MoE 计算的主路径，四者对当前层输出依赖明确，整体上仍然是串行为主。
5. 到下一层时，如果上一层 prefetch 成功，则下一层会在 `availability check` 后直接命中 swap buffer，减少甚至避免新的 CPU miss load。也就是说，跨层重叠主要来自“当前层 compute”和“下一层专家预取”之间，而不是两层 attention 彼此并行。
6. 从平均时间轴看，`next-layer prefetch` 的结束时间通常晚于当前层 `scatter` 的结束时间，说明它经常跨过当前层的后半段，进入下一层开始之前的一小段窗口；这正是它能提高 combined hit rate 的来源。

