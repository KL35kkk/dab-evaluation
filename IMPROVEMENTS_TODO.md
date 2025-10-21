Rule-Based 评估

evaluation/hybrid_evaluator.py:114-126 中直接使用 response_lower 但未定义，一旦命中 expected_answer 或 context 分支就抛出异常，导致评分回落到 0 或整个 hybrid 流程提前失败，准确率直接归零。
规则只设加分不设扣分：匹配到几个关键词就能轻松 ≥0.2，即便回答事实错误也不会被惩罚，造成大量假阳性。
expected_answer 仅做子串匹配，忽略大小写之外没有任何同义词/格式归一化处理；只要答案写法稍有变化（数值加单位、时间格式不同等）就匹配失败，漏判率高。
缺少对 agent_response 的基本清洗：没有统一大小写、去除标点或格式归一化，导致对“0xabc” vs “0xABC” 这类等价信息判定不一致。
规则覆盖面窄且与问题类型无关，例如技术类提问却只依据通用关键词，说明质量好坏和得分关联度弱。
优化建议：
- 执行统一预处理：生成 response_normalized、expected_normalized，用于大小写、标点、数值格式、地址格式的归一化，可复用 EnhancedScoring 的 normalize 功能。
- 在规则中引入扣分逻辑，对检测到的否定词、错误事实（如错误的合约地址、偏差过大的数值）以及明显敷衍的回答进行惩罚，平衡 precision/recall。
- 基于问题类别拆分规则集（数值、步骤、比较等），先判别 question 类型，再执行对应的关键词或结构化比对。
- 将 expected_answer 匹配升级为 token 级或模糊相似度匹配，引入 difflib、Levenshtein 或 n-gram 比对，降低格式差异带来的漏判。
- 提取结构化关键字段（0x 地址、日期、百分比等）并显式比较结果，提升对关键信息的鲁棒性。
增强评分集成

即使启用 EnhancedScoringSystem，score_answer 的语义相似度仅靠词汇集合和词序（_calculate_semantic_similarity），对短句或同义表达极不敏感；很多真实正确答案会被打低分。
_select_optimal_scoring_method 只靠有限关键字判断类别，遗漏如 “which/where/how many”等常见问法，方法选择容易错判，导致用错评分策略。
_calculate_dynamic_weights 用 expected 文本推断题型而非 question，若参考答案是数值/地址本身不含关键字，就无法切换到格式/事实权重；同一问题的不同参考答案会触发不同权重，结果不稳定。
没有把 ScoringResult.confidence 等信息纳入最终加权，增强评分得分和置信度之间不做调节，造成噪声大的评分也会被等权采纳。
优化建议：
- 扩充问题意图分类器：综合 question、context、expected 三端信息并覆盖更多 interrogatives（which、where、how many、compare 等），必要时引入轻量 ML 模型。
- 替换或补充语义相似度计算，尝试使用本地向量模型（sentence-transformer、FastText）并缓存 embedding，减轻对词面重叠的依赖。
- 结合 ScoringResult.confidence 重新计算 rule_score = score * confidence，在置信度偏低时自动降权或触发回退。
- 利用 breakdown 各项低分信号驱动后处理，例如 factual < 0.4 时直接降级评定或附加错误标记。
- 构建增强评分的熔断机制：当内部抛异常或置信度低于阈值时，回退到改进后的规则引擎并记录原因。
Hybrid 汇总

llm_evaluation_threshold 从未使用（evaluation/hybrid_evaluator.py:16-19），无论 LLM 分数多低都会参与混合，阈值设定失效。
权重固定 0.3/0.7，且不随输入质量或置信度调整；一旦 LLM 评估不稳定（详见下节），整体分数被强行拉向错误方向。
异常处理只是在失败时把 LLM 分数置 0，但不会记录失败原因或调整权重，高误差场景下仍然用 0.7 的系数计算，导致混合评分长期低估。
优化建议：
- 落实 llm_evaluation_threshold：设计分段或阈值逻辑，低于阈值时直接忽略或按比例衰减 LLM 权重。
- 引入自适应加权，结合 rule/LLM 的历史表现、置信度或题型建立简单的权重学习器（线性回归、贝叶斯更新等）。
- 在 details 中记录每次混合决策的原始输入、权重与最终输出，方便离线调参和问题追踪。
- 针对异常进行分级：网络失败、解析错误与置信度不足分别处理，连续异常触发熔断以保护整体分数。
- 暴露诊断指标（例如 LLM 调用耗时、失败率、降级次数）给监控系统，及时捕捉模型端问题。
LLM 评估

dab_eval/evaluation/llm_evaluator.py:46-64 若返回非 JSON，就简单按回复长度给 0.7/0.5，基本无判别力；很多明显错误的评语只要字数够多就高分。
对启用/禁用并没有基于模型置信度或提示格式验证，只要模型输出 JSON，分数就被直接采纳，没有最基本的数值范围校验或一致性检查。
未结合 llm_evaluation_threshold 做降权/弃用，也没有基于 reasoning 的 sanity check，导致 LLM 系统性偏差无法被发现。
优化建议：
- 约束输出 Schema，要求返回 {"score": float, "confidence": float, "reasoning": str, "flags": [str]}，解析失败立即降级并记录。
- 针对 score/ confidence 设置范围校验与平滑策略（如温度缩放或历史均值校准），防止离群值直接写入最终分数。
- 构建 reasoning 审核规则，检测否定语句、推测性词汇或自我否定，触发扣分或人工复核。
- 在 config 中支持可插拔 LLM 客户端（OpenAI、Azure、本地模型等），并建立 A/B 测试脚本，定期比较不同模型的评估一致性。
- 记录原始 LLM 输出、解析结果与最终决策，形成审计日志，为回归分析与模型更新提供依据。
综合影响

规则层面不可用（变量未定义）直接让混合框架频繁降级，增强评分/LLM 任何一侧稍出问题整体就崩。
加权策略缺乏动态调节，任意一侧漂移都会放大误差；缺乏负面激励又没有阈值，导致高假阳性和假阴性并存。
语义与格式判定都以简单 heuristics 代替，面对真实任务的多样表达无法稳健工作，精准度和召回率会双双受损。
如果需要进一步提升准确率，可以先补齐变量/归一化等基础漏洞，再讨论如何加权和增强语义判定。随时告诉我下一步想优先优化哪部分。
验证与监控

- 构建覆盖主要题型的标注集（建议 ≥200 条），用于回归测试和指标对比，确保每次改动对关键指标的影响可衡量。
- 为规则、增强评分、混合逻辑和 LLM 解析分别编写单元测试，覆盖典型边界场景（空输入、格式异常、冲突信息等）。
- 执行离线评测脚本，统计 accuracy、precision、recall、F1 及置信度分布，并将最佳阈值记录在文档中。
- 输出结构化日志（JSON lines），包含 question、rule_score、llm_score、confidence、决策原因等字段，便于后续数据分析与报警。
- 设定上线门槛（例如准确率较基线提升 >5%，且 precision/recall 均高于 0.7），作为发布前的验收标准。
