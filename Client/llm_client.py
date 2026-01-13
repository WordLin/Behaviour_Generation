import os

import streamlit as st
from openai import OpenAI
from pathlib import Path


# ===== 1) 基本配置 =====
API_KEY = os.getenv("OPENAI_API_KEY")  # 按你的要求，直接写入代码
MODEL = "gpt-4o-mini"           
PROMPT_PATH = r"C:\AgentProject\Behaviour_Generation\behaviour_prompt\COM-B.txt"
OUT_JSON = "method_cards_output.json"

# ========== 初始化 OpenAI 客户端 ==========
client = OpenAI(api_key=API_KEY)

def load_prompt():
    path = Path(PROMPT_PATH)
    if not path.exists():
        st.error(f"Prompt 文件不存在: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def call_llm(user_input: str, base_prompt: str) -> str:
    """使用 Responses API 调用 LLM"""
    full_prompt = f"{base_prompt.strip()}\n\n用户输入：{user_input.strip()}"
    try:
        response = client.responses.create(
            model=MODEL,
            instructions="你是一个行为规划助手，请根据COM-B模型和干预功能分析用户的输入。",
            input=full_prompt,
            temperature=0.2,
        )
        return response.output_text
    except Exception as e:
        return f"[ERROR] {e}"

# ========== Streamlit 前端 ==========
st.set_page_config(page_title="COM-B 行为分析助手", page_icon="🧠", layout="centered")

st.title("🧩 COM-B 行为分析助手")
st.markdown("""
输入你当前的行为目标、困惑或想改善的生活习惯，我将根据 **COM-B 模型** 自动分析：
- 你的能力（Capability）、机会（Opportunity）、动机（Motivation）  
- 可能的干预路径（Education, Training, Environmental Restructuring 等）  
- 并生成初步行动建议
""")

# 用户输入框
user_input = st.text_area(
    "请输入你的目标或困惑：",
    placeholder="例如：我想提高睡眠质量，但总是拖到很晚才睡。",
    height=120
)

if st.button("开始分析", type="primary"):
    if not user_input.strip():
        st.warning("请输入内容。")
    else:
        with st.spinner("正在分析中，请稍候..."):
            base_prompt = load_prompt()
            if not base_prompt:
                st.stop()
            result = call_llm(user_input, base_prompt)
        st.subheader("🔍 分析结果")
        st.write(result)

         # 结果保存
        out_path = Path("streamlit_output.txt")
        out_path.write_text(result, encoding="utf-8")
        st.success(f"✅ 结果已保存到本地文件：{out_path.resolve()}")

st.markdown("---")
st.caption("Powered by OpenAI Responses API · COM-B 行为模型分析 · Streamlit 前端展示")
        
