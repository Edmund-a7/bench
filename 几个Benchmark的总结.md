1.VBench
标准的单视频计算指标，包括Total、Quality、Semantic，但仅支持使用VBench的 prompt 作为输入的结果
●Quality：
○subject_consistency
○background_consistency
○temporal_flickering
○motion_smoothness
○aesthetic_quality
○imaging_quality
○dynamic_degree
●Semantic
○object_class (物体类别)：GRIT 生成视频帧的密集描述，检测描述中是否包含目标物体，计算成功率 = 包含物体的帧数 / 总帧数
○multiple_objects (多个物体)：GRIT 生成描述，检测描述中“and”连接的多个物体，计算所有物体同时出现的帧数比例
○human_action (人类动作)：使用动作识别模型分析视频中的人类动作，与提示中的动作描述相匹配
○color (颜色)：GRIT 生成描述，检查指定物体的颜色属性，计算帧数比例
○spatial_relationship (空间关系)：GRIT 检测物体边界框，计算中心点坐标，判断空间关系，计算空间关系匹配分数
○scene (场景)：Tag2Text模型生成场景标签，检查标签是否包含目标场景，计算匹配成功率
○appearance_style (外观风格)： CLIP 特征和提示中风格描述的相似度分数
○temporal_style (时间风格)：ViCLIP特征和提示中时间风格描述的相似度分数
○overall_consistency (整体一致性)：ViCLIP特征和完整提示的相似度分数
●Total：Quality和Semantic的加权平均

2.VBench-Long
普通框架评估2-8s，VBench-Long提出快慢评估方法，扩展到任意长度视频
●现有的一致性指标主要衡量帧与帧之间的相似度，虽然能捕捉局部的时间连贯性，但无法检测在更长时间尺度上累积的偏移或不一致问题
●VBench-Long将评估分解为两个分支：
○慢分支（片段内）：在短视频片段中进行高帧率评估，捕捉细粒度的时间一致性
○快分支（片段间）：跨片段进行低帧率评估，通过比较每个片段的代表性帧来捕捉长程一致性
通过语义场景检测和固定时长分割，自动将长视频拆分为易于处理的片段，然后利用基于分位数的分布对齐来融合两个分支的分数。

支持的维度：VBench-Long 目前支持以下用于长视频评估的一致性和质量维度：
●subject_consistency, background_consistency
●motion_smoothness, dynamic_degree
●aesthetic_quality, imaging_quality
●temporal_flickering (with static scene filtering)
但对自定义的视频，仅支持以下维度：
●subject_consistency：DINO 变换，提取特征后，计算当前帧与前一帧/第一帧的余弦相似度后取均值
●background_consistency：CLIP 变换，提取特征后，计算当前帧与前一帧/第一帧的余弦相似度后取均值
●motion_smoothness：每隔1帧采样一次，使用AMT 模型插值，计算原始中间帧与模型生成中间帧的L1距离，取均值；
●dynamic_degree：按FPS≤8 采样，使用RAFT 计算相邻帧的光流幅度，设置阈值，如果超过阈值的帧对数量达到要求，判断为动态视频；
●aesthetic_quality：CLIP 变换，提取特征后，使用 LAION 预测0-10 的美学分数，取均值
●imaging_quality：缩放、归一化后，使用MUSIQ模型评分，取均值

视频预处理：
●根据语义检测场景：使用PySceneDetect 搭配 ContentDetector，在语义边界（场景切换处）分割长视频
●根据固定长度分割：根据使用模型的预训练视频长度，将视频分割为不同短片段
○subject_consistency: 使用 DINO，2s
○background_consistency: 使用 CLIP，2s
○human_action: 使用 UMT，4s
○overall_consistency: 使用 ViCLIP，4s

慢分支：In-Clip Consistency
●标准VBench 指标，独立评估每个片段的分数，最终求均值

快分支：Clip-to-Clip Consistency
●从每个片段中采样第一帧，拼接成一个新视频进行评估。
●subject_consistency：使用DINOv2, DreamSim，
●background_consistency：使用CLIP, DreamSim，

融合快慢分支：
●由于评估方法和特征提取器的不同，它们具有不同的分布
●在融合前使用分位数映射，将片段间分数分布与片段内分数分布对齐

分位数映射与分数融合：
●将快分支评分分布对齐到慢分支评分分布：先把快分支评分排序，对每个视频转换为分位数，再对慢分支评分构建CDF，最后建立每个快分支分位数和慢分支值的对应关系，即映射表；实际上是使用大量数据预计算好的
●将慢分支分数和映射后的快分支分数加权作为最终分数

特殊情况：Temporal Flickering with Static Filtering时间闪烁与静态过滤
●闪烁是动态视频中的常见问题，例如同一主体在前后帧之间亮度发生跳变
●而静态场景的编码噪声也可能导致亮度发生变化，但常被误判为闪烁，因此先通过RAFT 计算光流，根据分数剔除静态视频，仅对动态场景进行闪烁评估

模型：
●DINOv2: 2023.4，
●CLIP: 2021.1，
●ViCLIP：2023.7，
●GRIT:2022.7，
●Tag2Text(Recognize Anything):2023， 
●DreamSim：2023.6，
●MUSIQ: 2021，
●LAION: 2022，
●UMT: 2023，
●AMT: 2023.6，
●RAFT: 2020，

3.MemFlow
●对于交互生成60s 的VBench-Long指标，报告了Subject Consistency、Background Consistency、FPS、CLIP Score
●对于单 prompt 的 5s/30s 的VBench-Long指标，报告了Total Score、Quality Score、Semantic Score、FPS
其中
●CLIP_Score：在每一个 10s 的区间内，计算该视频片段与其对应的文本 Prompt 之间的 CLIP 特征余弦相似度
●Total Score、Quality Score、Semantic Score仅支持用VBench 本身的 prompt
●Subject Consistency、Background Consistency支持自定义，所以我们专门做分段长视频的应该好一点

4.IVEBench
(1)Video Quality
●VTSS：监督模型，综合评估了构图、美学、清晰度、色彩饱和度和内容自然度 。这是对视频“卖相”的整体打分
●Subject Consistency
●Background Consistency
●Temporal Flickering
●Motion Smoothness
(2)Instruction Compliance
●OSC: 计算目标视频与**目标提示词（Target Prompt）**之间的 VideoCLIP-XL2 语义相似度。关注整体场景是否变了 。
●PSC: 计算目标视频与**目标短语（Target Phrase）**之间的 VideoCLIP-XL2 相似度。关注具体的编辑操作（如“把红车变成蓝车”）是否达成 。
●IS：对于“运镜”、“主体动作”等难以用向量相似度衡量的任务，作者使用 Qwen2.5-VL，输入指令和视频，让模型进行5分制打分。这模拟了人类的主观判断 。
●QA：针对数量编辑（如“增加一只鸟”）。使用 Grounding DINO 检测物体数量，对比指令要求，判定正误（1或0）
(3)Video Fidelity
●SF (Semantic Fidelity)：计算源视频与目标视频的 VideoCLIP-XL2 特征相似度，确保整体语境未丢失 
●CF (Content Fidelity)：针对运镜或角度变化导致像素无法对齐的情况，使用 Qwen2.5-VL 评估非编辑区域的内容是否被正确保留（5分制）
●MF (Motion Fidelity)：使用 CoTracker3 提取长时运动轨迹，通过匈牙利算法匹配源和目标视频的轨迹点，计算位置和速度的距离。如果轨迹变了（例如原本挥手变成了拍手），得分就会降低。

5.LV-Bench
引入VDE 分数
●将长视频均匀分段，每段通过特定方法打分
●计算各段分数相对首段分数的相对变化率，最后按线性递减权重加权求和作为分数
●即越早发生的漂移，收到的惩罚越重
设计五个指标：
●Clarity: 使用拉普拉斯算子计算亮度方差，检查画面是否逐渐变模糊
●Motion: 计算光流向量的L2 范数，检查运动幅度是否突变
●Aesthetic: 预训练美学评分模型，检查美学分数
●Background: 语义分割获取背景掩码，计算区域内静止像素的比例，检查背景是否存在"闪烁"等不正常的变化
●Subject: 计算各帧主体的特征向量与第一段主题的平均特征之间的余弦相似度
○在每帧中获取包含主体的图像区域，编码特征向量，计算余弦相似度
6.我们的
针对流式交互的长视频这块，每段prompt都会在前一段 prompt 上做一定修改
也可以分Quality, Temporal Consistency, Semantic Alignment

VDE 和 VBench-Long 的相同点：
●都是在短视频打分的基础上，增加了长视频的融合方法
●VDE 是根据分数相对变化率，线性衰减加权求和作为分数
●VBench-Long 是抽帧组成新视频，通过分位数加权求和作为分数
