## 行为生成项目说明

面向行为科学与健康干预的两个子模块：
- **COM-B 行为分析助手**：基于 COM-B 模型和干预功能的提示词，通过 Streamlit 提供交互式分析与建议输出。
- **网页爬取与方法卡生成器**：爬取健康/饮食类网页，抽取可执行习惯建议，按自定义 JSON Schema 生成/合并方法卡（method_template），并落盘为 JSONL。

## 目录结构
- `Client/llm_client.py`：Streamlit 前端与 OpenAI 调用；读取 `behaviour_prompt/COM-B.txt` 提示词，输出保存到 `streamlit_output.txt`。
- `behaviour_prompt/`：行为分析与 BCW 轮提示词（`COM-B.txt`、`rechange_wheel.txt`）。
- `web_crawler/main_crawl.py`：命令行入口，加载 Schema、调用爬取与卡片生成。
- `web_crawler/Crawl_test01.py`：核心逻辑（抓取、正文抽取、候选挖掘、规则/LLM 解析、Schema 校验、去重与落盘）。
- `web_crawler/cards_schama.json`：方法卡与内容卡的 JSON Schema。
- `AtomCard_data/cards_params_example.json`：示例方法卡与内容卡数据。
- `web_crawler/*.jsonl`：示例/历史运行产物。
- `运行脚本.txt`：示例运行命令与环境变量设置。

## 环境准备
1) Python >= 3.10（建议 3.11）。  
2) 安装依赖：
```bash
pip install -r requirements.txt
# 代码还依赖 jsonschema；可选 trafilatura 提升正文抽取
pip install jsonschema trafilatura
```
3) OpenAI Key：代码当前在 `Client/llm_client.py` 硬编码了示例 Key，建议改为环境变量：
```bash
setx OPENAI_API_KEY "your_key"
```
并在代码中将 `API_KEY = os.environ["OPENAI_API_KEY"]`，或手动替换为你的有效 Key。

## 使用方法

### 1) 启动 COM-B 行为分析助手
```bash
streamlit run Client/llm_client.py
```
浏览器输入目标/困惑，返回 COM-B 分析与建议，并将结果写入 `streamlit_output.txt`。如提示词路径为绝对路径（默认写死在代码中），请改为本仓库内相对路径，例如：
```python
PROMPT_PATH = "behaviour_prompt/COM-B.txt"
```

### 2) 网页爬取并生成方法卡
示例命令（来自 `运行脚本.txt`），会将产物写入 JSONL：
```bash
python web_crawler/main_crawl.py \
  --url "https://www.nhs.uk/live-well/eat-well/how-to-eat-a-balanced-diet/how-to-cut-down-on-sugar-in-your-diet/" \
  --schema_path web_crawler/cards_schama.json \
  --out_jsonl web_crawler/cards.jsonl \
  --locale en_US
```
参数说明：
- `--url`：目标网页（主要针对英文饮食/控糖内容）。
- `--schema_path`：校验用 Schema，默认 `web_crawler/cards_schama.json`。
- `--out_jsonl`：输出/合并文件；同 ID 会合并证据。
- `--locale`：卡片 locale（默认 `en_US`）。
- `--no_llm`：仅走规则匹配，跳过 LLM 兜底。

流程概要：
1) 抓取 HTML (`requests`) → 正文抽取（优先 `trafilatura`，回退 `bs4`）。  
2) 候选挖掘：通过正则/关键词筛高召回的行动语句。  
3) 质量门控：过滤噪声标题/导航等。  
4) 快速规则映射：针对高频饮食模式（替换含糖饮料、低糖早餐等）直接生成最小方法卡。  
5) 若规则未命中且允许 `--no_llm` 未开启，则调用 OpenAI 结构化输出生成卡片。  
6) Canonicalize + Schema 校验 + 去重/合并（同 ID 合并证据、取高置信度），最终原子重写 JSONL。失败记录写入 `<out_jsonl>.failures.jsonl`。

## 数据与示例
- `AtomCard_data/cards_params_example.json`：包含一个方法卡与两个饮品食谱内容卡示例，可作为 Schema 理解与 UI 展示参考。  
- `web_crawler/cards.jsonl`、`cards_1stRun.jsonl` 等：历史抓取产物，可直接用作样例或继续合并。

## 注意事项与改进建议
- **安全**：仓库中存在示例 OpenAI Key，生产使用前务必替换/改为环境变量，避免泄漏与额度风险。  
- **Schema 依赖**：`jsonschema` 未列入 `requirements.txt`，请确保安装。  
- **抓取合法性**：生产环境应增加 robots.txt 检查、限速与缓存策略。  
- **模型与提示词路径**：`llm_client.py` 默认使用 `gpt-4o-mini`，请根据账号配额调整；同时修改 `PROMPT_PATH` 为本地相对路径以便跨机运行。  
- **trafilatura 可选**：未安装时自动回退 bs4，但正文质量可能下降。

