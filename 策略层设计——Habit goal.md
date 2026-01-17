# 策略层设计——Habit goal

## 基础盘分析

- **目标**
  - 当用户进入Habit goal的意图之后，目的是建立最小习惯，检索行为原子库，可适应的调整，最后编排习惯进行打卡。最终都要收敛到一套行为原子库中。因为行为原子库的构建十分重要。此处采用戒糖/代谢作为wedge数据然后横向扩展。
- **思考难点**
  - 以什么情境触发（When/Where）
  - 以多大剂量（How much）
  - 如何降低摩擦（Make it easy）
  - 如何追踪与反馈（Measure/Reward）
  - 如何在失败后自适应调整（Adapt）------> 进入troubleshooting意图
  - prompt大量使用的幻觉和成本上升
  - Habit goal需要**Behavior specification（行为规格化）**
    - 具体到：每天几次、每次多少、在什么时候、以什么触发器发生、怎么最省力完成。

- **解决方法**
  - **行为科学 Prompt 分析**
    - 对齐：可以先把用户句子规范到可执行的行为描述（推荐字段：动词、对象、剂量、频率、情境、约束、偏好）。
    - COM-B 诊断 + BCT 处方：主要用来习惯遇到障碍怎么继续，障碍在什么地方的分析
    - **把“文本理解”变成“可计算的状态”**，后面的检索、排序、学习才能做。
  - **原子行为库怎么定义**
    - 参数怎么定义
      - 干预函数
      - 障碍缺口
      - 失败回退
      - 剂量
      - 追踪
    - **反向工程**
      - 通过BCT/COM-B 标签、适用情境、难度、禁忌、预期指标把行为原子卡补充，做成极高质量的数据资产，在用户路由阶段可以通过多级别的过滤减少prompt成本，加快效率。
      - 长期方案可以用数据把“分析器”蒸馏成小模型（护城河）。
  - **组合与排程**
    - 输出不是“1 张卡”，而是：1 张**主卡**（最小可执行，确保启动），1–2 张**增强卡**（可选升级，形成梯度），1 张**失败兜底卡**（Plan B：缩小剂量/替代行为/改提示）。
  - **持续更新JITAI**
    - **信号采集**：完成/跳过、完成时刻、连续性、失败原因（可选传感器：步数/睡眠等）
    - **状态估计**：用户当前“可行性/动机/机会”状态（可用简单 HMM/贝叶斯更新/或规则先做）
    - **策略选择**：早期用启发式：失败→减小剂量、换 cue、换环境改造卡，中后期上 **Contextual Bandit**。
    - **卡库更新**：低表现卡降权、合并或重写步骤，高频失败情境生成“新卡/替代卡”。并进入人工审核队列
  - **垂域数据收集**
    - **分析指标**
      - **需求强度（Pain & urgency）**：用户是否“不得不做”，失败代价是否高。
      - **支付能力与支付意愿（WTP）**：B2C 是否能到 $20-$150/月，或是否存在 B2B payer（雇主/保险/医疗）。
      - **数据丰富度（Data richness）**：是否天然有可持续、客观、可自动采集的信号（可穿戴/检测/平台日志）。
      - **效果可衡量（Measurability）**：是否能在 2–12 周内看到客观指标改善（A1C、体重、步数、分数、产出等）。
      - **可标准化可复制（Standardization）**：是否能沉淀“原子行为卡”并跨人群复用（而不是强依赖人工教练）。
      - **增长与获客渠道（Channel）**：是否有清晰可规模化渠道（医生/雇主福利/社区/考试节点/SEO）。
    - **选择**
      - 减脂/控糖（代谢健康）——最符合“强需求 + 高价 + 数据丰富”
  - **数据对象：BehaviorPlan**
    - **分析**
      - Outcome goal需要补一个步骤：**Goal → Behavior decomposition（目标到行为分解）**。这一步不是“心理分析”，而是把结果目标转成可执行的行为集合，并明确衡量方式（否则无法闭环）。Habit goal需要做的不是“诊断动机”，而是补齐 **Behavior specification（行为规格化）**。
      - 两者最终都必须落到一个统一的数据对象：**BehaviorPlan = {行为卡集合 + 触发器 + 测量指标 + 自适应策略}**
         然后进入同一个闭环：**选择卡 → 执行 → 观测 → 归因/调整 → 更新策略与卡库**。
    - **解决方案**
      - 对用户指令抽取/推断最关键槽位（不足就补问一个最小问题）
      - 观测的主指标询问
        - 我怎么知道要问用户含糖饮料次数/甜食加餐次数/外卖甜品次数这些问题，当场景变化时怎么做到能询问相关的问题？依靠prompt吗
          - **两段式推理**：先小模型/规则做粗识别 + 置信度；低置信度才上大模型。
        - 怎么从用户的指令中来确定剂量边界？蔬菜剂量，跑步剂量等等？
          - **RAG + 程序化营养计算**：让 LLM 只做“识别与结构化选择”，不要做“营养学自由发挥”。剂量由数据库挖掘来控制。
      - 生成原子行为卡
      - 自适应闭环（避免“死板打卡”）



- **最小可行方案**

  - **LLM调用分析**

    - 在前期明确一个点就是需要合理判断什么时候调用LLM，而不是避免调用或者大量调用。需要从从“API 依赖”走向“自有模型覆盖大头”。这是很多 AI 应用从 0→1→10 的成本演进路线（概念来源可以追溯到经典的**知识蒸馏/teacher-student**范式）：
      - 用大模型（外部 API）在早期快速做出产品与数据闭环
      - 积累真实世界的带反馈数据（他们隐私政策也提到会用匿名/聚合图片改进模型）
      - 把大模型当 teacher 产生结构化标签/解释
      - 训练/微调自家更小的视觉模型与分类器，覆盖 80–95% 的常见场景
      - 只把长尾复杂样本回退给大模型
    - 这条路线的关键优势是：**把边际成本从“每次调用计费”变成“固定算力 + 更低的单次推理成本”**，并且可控性更强（延迟、稳定性、合规）。

  - **重复习惯/数据的缓存**

    

### 1. 任务型对话实现方法

- **步骤总结**
  1. **行为模板化**
  2. **最小可执行槽位**
  3. **问询策略**
  4. **低置信度启用LLM API**
  5. **RAG 检索行为原子库**



### 2. 行为原子库

- 新问题：
  - 当前是用很小的行为习惯作为单位构建原子行为卡，但是如果行为原子卡太小，比如减脂习惯中的早餐存在多种不同的饮食组合，每天都不同，那么存在多张不同的早餐饮食卡作为最小的原子卡，比如3 egg whites and 1 whole egg, scrambled 2 slices of whole grain bread (100% whole wheat, rye, oat or gluten-free bread) ½ cup cooked spinach ¼ cup low-fat shredded cheese作为一个原子卡。但是如果用户指令只是喝一杯蔬菜汁，多吃水果的习惯就无法使用这种原子卡，比如单独准备多种蔬菜汁的配方作为多张不同的原子卡，这样的话就存在大量的数据开销。我应该怎么选择行为原子卡的最小单位，才能尽量覆盖不同的场景和需求？

- **采用构建卡包的方式**
  - 单张原子卡很难同时承载“主动作、提示、环境重构、监测、反馈”。但行为科学上有效干预通常是**组件组合**（多个 BCT 叠加）。除了 `BehaviorAtomCard`，再设计一个组合对象。
  - **MethodBundle（方法包）**
    - bundle_id
    - `goal_tags`：例如 reduce_added_sugar / eat_more_veg
    - `cards`：主卡 + 提示卡 + 环境卡 + 监测卡 + 兜底卡（每张都是原子卡）
    - `recommended_sequence`：先环境后行为？先监测后替换？
    - `adaptation_rules`：失败 2 次降级；连续成功 5 次升级；换触发器优先级等

- **单卡关键参数有哪些**
  - 改善什么
  - 时间
  - COB行为标签
  - 频率
  - 剂量
  - 替换方案
  - 前置要求
  - 安全风险



- **字段分层构建行为原子卡**
  - 核心实体：BehaviorAtomCard（原子行为卡）
    - **基础信息**
      - `card_id`：稳定 ID（不可随文案变化）
      - `version`：语义版本（内容或规则变化就升版本）
      - `status`：draft/active/deprecated
      - `domain`：如 metabolic_health / study / productivity
      - `locale`：语言、地区差异（饮食/单位/合规）
    - **行为规格（必须，可检索）**
      - `behavior.action_verb`：walk / prepare / choose / replace / log…
      - `behavior.object`：vegetables / sugary_drink / fruit…
      - `behavior.dose`：数值+单位（minutes / servings / grams / times）
      - `behavior.frequency`：daily/weekly 或 RRULE（工程更稳）
      - `behavior.intensity`：可选（轻/中/高）
    - **触发器（强烈必须）**
      - `steps`：最多 3 步（减少认知负担）
      - `prep_required`：是否需要准备（食材/装备/场地）
      - `time_cost_estimate`：预计用时
      - `effort_level`：1–5
      - `alternatives`：Plan B（更小剂量/替代地点/替代动作）
  - **机制标签层**
    - COM-B 缺口标签（强推）
    - BCW 干预功能（强推）
    - BCTTv1 标签（强推，库的“原子干预组件”）
    - Fogg 失败定位标签（轻量但非常实用）
  - **适用性与约束层：防止“推荐正确但不可执行”**
  -  测量与学习层：闭环与可持续优化的根基



### 3. 内容爆炸怎么处理

- 解决思路不是继续把卡做得更小，而是把“最小单位”从 **内容实例（recipe instance）** 上移到 **可复用的行为方法（method template）**，然后把配方/食材作为“可插拔内容资源”去填充。
- **行为原子卡**
  - 原子卡定义成“决策点”。需要具备泛化性，像总结性的知识点一样。当用户在什么时候需要做决定？决定的动作是什么？该卡就必须匹配当前场景，然后才是内容实例的组合出现。
  - 最小单位可以定义为
    - **Atomic Method Card =（行为意图 + 触发器 If–Then + 剂量策略 + 约束/可行性 + 兜底方案 + 评价指标）**而不是某一道具体早餐/某一杯具体蔬菜汁。
    - 具体的配方可以作为一个插拔的组合，自由组合。属于 **Content Variant（内容变体）**，是可替换填充物，不应作为原子卡本体。
  - **目标：实现“1张方法卡 + 多个配方轮换”**





## 数据挖掘策略

### 1. 步骤

- 数据的schema已经写好了，包含 **method_template 和可插拔的内容变体外挂 content_variant schema** 。
- 选择一个允许抓取的网页 URL
- 调用 `ingest_url_to_cards(url, "schema.json", "cards.jsonl")`
  - **要点**
    - schema必须和结构匹配不然schema gate 直接拒绝
    - Structured Outputs 对 JSON Schema 支持的是子集，复杂 schema经常会导致请求被拒或输出难以严格受控
    - 这里采用LLM 先产出“最小可用结构”（MVP）→ 本地再用完整 schema 严格校验/再做 enrichment
- 输出的 `cards.jsonl` 里就是一条条通过校验的 `method_template` 卡



### 2. schema分析

- 基础数据类型

  - **`Id`**: 唯一标识符。格式要求：3-128 字符，只能包含字母、数字、下划线、连字符和点。

    **`Locale`**: 语言地区。格式严格固定为 `xx_XX`（如 `en_US`, `zh_CN`）。

    **`Tag / TagArray`**: 标签。用于分类和筛选，要求全小写。

    **`Dose`**: 剂量。包含 `value` (数值) 和 `unit` (单位，如 "meal", "glass", "min")。

    **`TimeHHMM`**: 时间格式。严格符合 `HH:MM`（24小时制）。

    **`TimeWindow`**: 时间窗口。包含两个时间点，表示开始和结束。

- **方法模板卡片 (MethodTemplateCard)**

  - **核心逻辑字段**

    - **`behavior_key`**: 行为的内部键名（如 `reduce_sugar_intake`）。
      - **`trigger`**: **触发器**。
        - `if`: 包含 `cue`（情境信号，如 "after_waking_up"）和 `time_window`。
        - `then`: 包含 `action`（要执行的动作标签）。

    - **`dose_policy`**: **剂量策略**。
      - `default`: 初始建议值。
        - `range`: 允许用户调整的最小值和最大值。
        - `upgrade_rule / downgrade_rule`: 难度进阶或降级规则（文案描述）。

  - **辅助与科学字段**

    - **`constraints`**: **约束条件**。包含所需器材、是否无糖、预计耗时、禁忌症标签。

      **`mechanism_tags`**: **行为机制**。

      - `comb_targets`: 基于 COM-B 模型，定义该习惯解决的是能力、机会还是动机。
      - `barrier_tags_solved`: 专门解决哪些心理或物理障碍。

      **`fallbacks`**: **备选方案**。当主要行为无法执行时（如出差、生病），用户可以做的替代方案。

      **`content_slots`**: **内容插槽**。

      - 这是一个占位符，定义了该习惯可以插入什么样的具体内容（如：这里需要一个“低糖食谱”）。
      - `candidate_query`: 定义了如何去筛选符合要求的内容变体。

- 内容变体卡片 (ContentVariantCard)

  - **`content_type`**: 内容类型。枚举值：`beverage_recipe` (饮品)、`recipe` (食谱)、`ingredient_option` (食材选项)、`prompt_copy` (提示文案)。

    **`recipe`**: **具体配方/步骤**。

    - `serving`: 一份的剂量。
    - `ingredients`: 列表，包含名称、数量和单位。
    - `steps`: 具体的执行步骤步骤列表。

    **`nutrition_estimate`**: **营养评估**。包含卡路里、膳食纤维、糖分、添加糖。

    **`bindings`**: **绑定关系**。

    - 这是最重要的字段，它定义了这个内容**可以被用在哪些 Method Template 的哪个 Slot 里**。通过 `method_template_id` 和 `slot_id` 建立链接。

- 溯源信息 (Provenance)

  - **`source_url`**: 原始网页地址。

    **`extracted_by`**: 提取者类型（人、规则解析器、LLM、混合模式）。

    **`confidence`**: 置信度（0 到 1 之间的浮点数）。

    **`evidence_spans`**: **证据原文**。从网页中直接摘录的原始文本，用于后期校验。

- 数据校验规则摘要 (Validation Gate)
  - **无额外属性**: `additionalProperties: false` 意味着 JSON 中不能出现 Schema 没定义的字段。
  - **必填项**:
    - `MethodTemplateCard` 必须包含 `trigger`, `dose_policy`, `content_slots`, `measurement` 等。
    - `ContentVariantCard` 必须包含 `recipe` 和 `bindings`。
  - **正则限制**: ID 和 Tag 的格式非常严格，不能有空格。



### 3. 爬虫代码框架

- **结构**

  - Schema 加载与校验：`load_schema` / `validate_or_raise`

    - `load_schema(schema_path)`：读取你定义的 JSON Schema（Draft 2020-12）。
    - `validate_or_raise(instance, schema)`：用 `Draft202012Validator` 对生成的 card 做校验，失败就抛异常。

  - 抓取网页：`fetch_html`

    - `requests.get(url)` 拉 HTML，带了 User-Agent，`raise_for_status()` 保证 4xx/5xx 直接失败。

  - 正文抽取：`extract_main_text`

    - **优先 trafilatura**：`trafilatura.extract(html, url=..., include_comments=False, include_tables=False)`
      - trafilatura 属于“正文抽取器”，核心思想是从 DOM/文本密度/结构特征中识别主内容区域（类似 Readability、Boilerpipe 那类工作）。
    - fallback：BeautifulSoup 粗抽

  - 候选挖掘：`mine_habit_candidates`

    - 整条链路里“高召回”的关键层，**启发式候选生成 + 打分过滤**
    - `_rebuild_blocks`：重建块
      - 把 text splitlines 后重新拼成块
    - `_split_sentences`：对长块做句切分
    - `_score_candidate`：多特征启发式打分
      - 结构信号：是否 bullet（+3）
      - 建议语气：ACTION_WORDS（+2）
      - 领域词：DIET_KEYWORDS（+1）
      - 餐次语境：MEAL_WORDS（+1）
      - “替换/交换”模式：REPLACE_RE（+2）
      - 定量指导：QTY_RE（+1）
      - avoid/reduce/limit + 目标词额外加分（+1）
      - 噪声词命中则减分（-6）
      - 最后以 `score >= 3` 作为通过阈值。

  - 结构化抽取：规则快路 + LLM 兜底

    - 规则快路：`fast_path_to_method_template_mvp`。命中后直接输出一个**最小、schema-conformant 的 method_template card**（MVP 版）。
    - LLM 兜底：`llm_extract_to_method_template`
      - 当规则不命中且 `use_llm_fallback=True`

  -  持久化：`append_jsonl`

    - 一行一个 JSON（JSONL），适合流式写入、增量入库、后续批处理。

  - 闭环编排：`ingest_url_to_cards`

    - 把所有步骤串起来：fetch_html → extract_main_text → mine_habit_candidates

  - 

    



### 4. 结果分析

- 失败率高分分析
  - 修改
    - 给 ingest 增加失败日志（否则无法定位 17/25 的真实原因）
    - 修 fast_path：不输出 `no_added_sugar: null`（否则必然有一批 validation fail）
    - 给 LLM 输出做后处理：强制 `content_slots=[]`（直到你有稳定 slot 抽取器）
    - 增加卡片级 fingerprint 去重与 evidence 合并
    - 重构 ContentVariantCard schema：按 content_type 分支 required


























































- 数据收集
  - https://arxiv.org/abs/2505.02851?utm_source=chatgpt.com 数据生成