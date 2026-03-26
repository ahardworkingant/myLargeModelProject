from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pydantic import Field, ConfigDict
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import MongoDBChatMessageHistory
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from calculate_data import medical_calculator

class MyFineTunedLLM(LLM):
    """封装本地微调模型的LangChain自定义LLM类"""

    # 声明模型属性
    model_name: str = Field(..., description="模型路径")
    model: Any = Field(default=None, description="模型实例")
    tokenizer: Any = Field(default=None, description="分词器实例")

    # Pydantic 2.x 配置
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, model_name: str, **kwargs):
        # 初始化基类
        super().__init__(model_name=model_name, **kwargs)
        # 在初始化时加载模型
        self._load_model()

    def _load_model(self):
        """加载模型和tokenizer"""
        print(f"Loading model from {self.model_name}, please wait...")
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return "my_fine_tuned_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """调用本地模型生成回复"""
        if self.model is None or self.tokenizer is None:
            self._load_model()

        # 构建模型输入格式
        formatted_prompt = f"问: {prompt}\n答:"

        # 将文本编码为token
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt")

        # 将输入移动到与模型相同的设备
        if torch.cuda.is_available():
            inputs = inputs.to(self.model.device)

        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2,
                top_p=0.8,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码生成的文本
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取回答部分（去掉原始提示）
        answer = generated_text[len(formatted_prompt):].strip()

        return answer

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

# 工具2: 医学计算器工具
def medical_calculator(query):
    """用于计算医学相关指标，如BMI、体表面积、肾小球滤过率(eGFR)等。输入应为需要计算的指标名称和数值。"""
    # 调用你的计算工具函数，这里需要你自行实现解析和计算逻辑
    return medical_calculator(query)

# 工具3: 通用聊天工具
def general_chat(query):
    """用于常规对话、问候、病情安慰或不需要专业知识的简单医学问答。"""
    # 直接调用我们微调好的模型
    response = llm.invoke(f"请以专业且友善的医生口吻回答以下问题：{query}")
    return response

# 将函数封装成LangChain Tool对象
tools = [
    Tool(
        name="Medical Calculator",
        func=medical_calculator,
        description="用于计算医学指标。输入应为计算请求，例如：'计算BMI，身高170cm，体重65kg' 或 '估算eGFR，肌酐100μmol/L，年龄45岁，女性'。"
    ),
    Tool(
        name="General Chat",
        func=general_chat,
        description="适用于问候、情绪安抚、非医学聊天或非常简单的健康建议。如果问题涉及复杂的专业知识，请优先使用知识库工具。"
    ),
]
# 为每个用户会话创建独立的agent和memory
def get_agent_for_session():
    # memory = get_memory(session_id)

    # 4. 初始化Agent
    # 定义系统提示，指导Agent的行为
    system_message = SystemMessage(content="""你是一个专业的、谨慎的医学AI助手。你必须严格遵守以下规则：
    1. 首先，仔细分析用户的问题属于哪一类。
    2. 如果问题涉及具体的医学专业知识（疾病、药物、治疗等），必须优先使用`Medical Knowledge Base`工具。
    3. 如果用户需要计算某项医学指标，必须使用`Medical Calculator`工具。
    4. 如果只是问候、闲聊或非常简单的问题，可以使用`General Chat`工具。
    5. 如果工具返回的结果不清楚或未找到答案，你可以基于自身知识（你的微调模型）进行回答，但必须声明“根据我的知识：”，并格外谨慎。
    6. 你的所有回答都必须基于可靠来源，不能捏造信息。对于超出能力范围的问题，应建议用户咨询专业医生。
    """)

    agent_kwargs = {
        "system_message": system_message,
    }
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # 这是一个非常强大的Agent类型，支持复杂的工具调用
        verbose=True,  # 打印调试信息
        # memory=memory,
        agent_kwargs=agent_kwargs,
        handle_parsing_errors=True,  # 优雅地处理解析错误
        return_intermediate_steps=True,
    )
    return agent

# 使用示例
if __name__ == "__main__":
    # 假设你微调后的模型保存在 './DeepSeek-R1-Distill-Qwen-7B' 目录下
    llm = MyFineTunedLLM(model_name='./medical_finetune_output/checkpoint-3000')

    while True :
        # 测试模型
        try:
            user_input = input("\n您: ")

            if user_input.lower() in ['退出', 'exit', 'quit']:
                print("对话结束")
                break
            elif user_input.lower() in ['清空', 'clear']:
                # dialogue_model.clear_history()
                continue

            agent = get_agent_for_session()
            result = agent.invoke({"input": user_input})
            # 生成回复
            # response = llm._call(user_input)
            print(f"AI: {result}")

        except KeyboardInterrupt:
            print("\n对话结束")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            # 可以选择清空历史或继续
            continue

# # 导入必要的库和模块
# from langchain.llms.base import LLM  # LangChain LLM基类
# from typing import Optional, List, Mapping, Any  # 类型注解支持
# from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face模型和分词器
# import torch  # PyTorch深度学习框架
# from pydantic import Field, ConfigDict  # 数据验证和配置管理
# from langchain.agents import AgentType, initialize_agent, Tool  # LangChain智能体相关
# from langchain.schema import SystemMessage  # 系统消息定义
# from langchain.memory import ConversationBufferWindowMemory  # 对话记忆管理
# from langchain_community.chat_message_histories import MongoDBChatMessageHistory  # MongoDB对话历史存储
# from langchain.chains import RetrievalQA  # 检索增强生成链
# from langchain_community.vectorstores import Chroma  # Chroma向量数据库
# from langchain_community.embeddings import HuggingFaceEmbeddings  # Hugging Face嵌入模型
# from calculate_data import medical_calculator  # 医学计算器工具
#
#
# # 自定义LLM类，封装本地微调模型
# class MyFineTunedLLM(LLM):
#     """封装本地微调模型的LangChain自定义LLM类"""
#
#     # 声明模型属性，使用Pydantic Field进行验证
#     model_name: str = Field(..., description="模型路径")  # 模型路径，必需字段
#     model: Any = Field(default=None, description="模型实例")  # 模型实例
#     tokenizer: Any = Field(default=None, description="分词器实例")  # 分词器实例
#
#     # Pydantic 2.x 配置，允许任意类型
#     model_config = ConfigDict(arbitrary_types_allowed=True)
#
#     # 初始化方法
#     def __init__(self, model_name: str, **kwargs):
#         # 初始化基类
#         super().__init__(model_name=model_name, **kwargs)
#         # 在初始化时加载模型
#         self._load_model()
#
#     # 加载模型和分词器的方法
#     def _load_model(self):
#         """加载模型和tokenizer"""
#         print(f"Loading model from {self.model_name}, please wait...")
#         try:
#             # 加载tokenizer
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 self.model_name,
#                 trust_remote_code=True  # 信任远程代码
#             )
#
#             # 加载模型
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_name,
#                 trust_remote_code=True,  # 信任远程代码
#                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # 根据GPU可用性选择精度
#                 device_map="auto" if torch.cuda.is_available() else None,  # 自动设备映射
#                 low_cpu_mem_usage=True  # 低CPU内存使用
#             )
#             print("Model loaded successfully!")
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             raise
#
#     # 返回LLM类型
#     @property
#     def _llm_type(self) -> str:
#         return "my_fine_tuned_llm"
#
#     # 调用模型生成回复的主要方法
#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         """调用本地模型生成回复"""
#         if self.model is None or self.tokenizer is None:
#             self._load_model()
#
#         # 构建模型输入格式
#         formatted_prompt = f"问: {prompt}\n答:"
#
#         # 将文本编码为token
#         inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
#
#         # 将输入移动到与模型相同的设备
#         if torch.cuda.is_available():
#             inputs = inputs.to(self.model.device)
#
#         # 生成文本
#         with torch.no_grad():  # 不计算梯度，节省内存
#             outputs = self.model.generate(
#                 inputs,
#                 max_new_tokens=512,  # 最大生成token数
#                 do_sample=True,  # 使用采样
#                 temperature=0.2,  # 温度参数
#                 top_p=0.8,  # 核采样参数
#                 eos_token_id=self.tokenizer.eos_token_id,  # 结束符ID
#                 pad_token_id=self.tokenizer.eos_token_id  # 填充符ID
#             )
#
#         # 解码生成的文本
#         generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#         # 提取回答部分（去掉原始提示）
#         answer = generated_text[len(formatted_prompt):].strip()
#
#         return answer
#
#     # 返回模型标识参数
#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
#         return {"model_name": self.model_name}
#
#
# # 工具2: 医学计算器工具
# def medical_calculator_tool(query):
#     """用于计算医学相关指标，如BMI、体表面积、肾小球滤过率(eGFR)等。输入应为需要计算的指标名称和数值。"""
#     # 调用医学计算器函数
#     return medical_calculator(query)
#
#
# # 工具3: 通用聊天工具
# def general_chat(query):
#     """用于常规对话、问候、病情安慰或不需要专业知识的简单医学问答。"""
#     # 直接调用我们微调好的模型
#     response = llm.invoke(f"请以专业且友善的医生口吻回答以下问题：{query}")
#     return response
#
#
# # 将函数封装成LangChain Tool对象
# tools = [
#     Tool(
#         name="Medical Calculator",  # 工具名称
#         func=medical_calculator_tool,  # 工具函数
#         description="用于计算医学指标。输入应为计算请求，例如：'计算BMI，身高170cm，体重65kg' 或 '估算eGFR，肌酐100μmol/L，年龄45岁，女性'。"
#         # 工具描述
#     ),
#     Tool(
#         name="General Chat",  # 工具名称
#         func=general_chat,  # 工具函数
#         description="适用于问候、情绪安抚、非医学聊天或非常简单的健康建议。如果问题涉及复杂的专业知识，请优先使用知识库工具。"
#         # 工具描述
#     ),
# ]
#
#
# # 为每个用户会话创建独立的agent和memory
# def get_agent_for_session():
#     # memory = get_memory(session_id)  # 获取对话记忆（已注释）
#
#     # 4. 初始化Agent
#     # 定义系统提示，指导Agent的行为
#     system_message = SystemMessage(content="""你是一个专业的、谨慎的医学AI助手。你必须严格遵守以下规则：
#     1. 首先，仔细分析用户的问题属于哪一类。
#     2. 如果问题涉及具体的医学专业知识（疾病、药物、治疗等），必须优先使用`Medical Knowledge Base`工具。
#     3. 如果用户需要计算某项医学指标，必须使用`Medical Calculator`工具。
#     4. 如果只是问候、闲聊或非常简单的问题，可以使用`General Chat`工具。
#     5. 如果工具返回的结果不清楚或未找到答案，你可以基于自身知识（你的微调模型）进行回答，但必须声明"根据我的知识："，并格外谨慎。
#     6. 你的所有回答都必须基于可靠来源，不能捏造信息。对于超出能力范围的问题，应建议用户咨询专业医生。
#     """)
#
#     agent_kwargs = {
#         "system_message": system_message,  # 代理参数
#     }
#
#     # 初始化代理
#     agent = initialize_agent(
#         tools,  # 工具列表
#         llm,  # 语言模型
#         agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # 代理类型
#         verbose=True,  # 打印调试信息
#         # memory=memory,  # 记忆（已注释）
#         agent_kwargs=agent_kwargs,  # 代理参数
#         handle_parsing_errors=True,  # 优雅地处理解析错误
#         return_intermediate_steps=True,  # 返回中间步骤
#     )
#     return agent
#
#
# # 主程序入口
# if __name__ == "__main__":
#     # 假设你微调后的模型保存在 './medical_finetune_output/checkpoint-3000' 目录下
#     llm = MyFineTunedLLM(model_name='./medical_finetune_output/checkpoint-3000')
#
#     # 持续对话循环
#     while True:
#         try:
#             # 获取用户输入
#             user_input = input("\n您: ")
#
#             # 退出条件
#             if user_input.lower() in ['退出', 'exit', 'quit']:
#                 print("对话结束")
#                 break
#             # 清空历史条件
#             elif user_input.lower() in ['清空', 'clear']:
#                 # dialogue_model.clear_history()  # 清空对话历史（已注释）
#                 continue
#
#             # 获取代理并处理输入
#             agent = get_agent_for_session()
#             result = agent.invoke({"input": user_input})
#             # 打印结果
#             print(f"AI: {result}")
#
#         # 处理键盘中断
#         except KeyboardInterrupt:
#             print("\n对话结束")
#             break
#         # 处理其他异常
#         except Exception as e:
#             print(f"发生错误: {e}")
#             # 可以选择清空历史或继续
#             continue