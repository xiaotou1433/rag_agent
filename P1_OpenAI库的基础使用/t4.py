from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

first_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你可以帮助起名字，仅回答名字，无需额外内容"),
        ("user", "我的邻居姓{lastname}，刚生了{gender}，请帮忙取一个名字，仅告知我名字，无需额外内容")
    ]
)

second_prompt = ChatPromptTemplate.from_messages(
    first_prompt.messages + [
        MessagesPlaceholder("name"),
        ("user", "请解析这个名字的含义")
    ]
)
model = ChatTongyi(model="qwen3-max")

chain = first_prompt | model | second_prompt | model
print(chain.invoke({"lastname": "张", "gender": "女儿"}).content)

