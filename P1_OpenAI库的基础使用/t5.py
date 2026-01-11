from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

# 2. 定义第一个 Prompt（取名）
first_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你可以帮助起名字，仅回答名字，无需额外内容，不要附带任何解释"),
        ("user", "我的邻居姓{lastname}，刚生了{gender}，请帮忙取一个名字，仅告知我名字，无需额外内容")
    ]
)

# 3. 定义第二个 Prompt（解析名字含义）
second_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你需要解析给定的中文名字的含义，包括字面意思、寓意、文化内涵等，回答清晰易懂"),
        MessagesPlaceholder(variable_name="name_messages"),  # 接收封装后的消息
        ("user", "请详细解析上面这个名字的含义")
    ]
)

# 4. 初始化模型和输出解析器
model = ChatTongyi(model="qwen3-max")
str_parser = StrOutputParser()

# 5. 构建单个完整链（核心：Lambda 函数完成中间数据转换和格式封装）
single_chain = (
    # 步骤1：执行取名 Prompt + 模型生成 + 提取纯名字字符串
        first_prompt
        | model
        | str_parser
        # 步骤2：Lambda 函数（中间转换）：将纯名字封装为 AIMessage 列表（符合 MessagesPlaceholder 要求）
        | (lambda name_str: {"name_messages": [AIMessage(content=name_str)]})
        # 步骤3：执行解析 Prompt + 模型生成最终解析结果
        | second_prompt
        | model
)

# 6. 调用单个链，完成完整需求
if __name__ == "__main__":
    # 直接 invoke 单个链，传入初始参数即可
    result = single_chain.invoke({"lastname": "张", "gender": "女儿"})

    # 输出结果
    print("完整结果（含消息格式）：")
    print(result)
    print("\n————————————————————")
    print("名字含义解析（纯内容）：")
    print(result.content)
