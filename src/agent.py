import openai
import tiktoken
import asyncio
from typing import List, Dict, Optional

class Agent:
    """
    一个通用的AI代理，能够与任何兼容OpenAI API的语言模型进行交互。
    """
    def __init__(self, name: str, model: str, api_key: str, base_url: Optional[str] = None, system_prompt: str = "", verbose: bool = False):
        """
        初始化Agent。

        Args:
            name (str): 代理的名称。
            model (str): 要使用的模型名称 (例如 "gpt-4-turbo")。
            api_key (str): 用于验证的API密钥。
            base_url (Optional[str]): 自定义的API端点URL。
            system_prompt (str): 分配给代理的系统级指令。
            verbose (bool): 是否启用详细输出模式。
        """
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.verbose = verbose
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _count_tokens(self, text: str) -> int:
        """使用tiktoken估算文本中的token数量。"""
        try:
            # "cl100k_base" 适用于 gpt-4, gpt-3.5-turbo, text-embedding-ada-002 等模型
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # 如果出现问题，回退到 gpt-2 的编码器
            encoding = tiktoken.get_encoding("gpt2")
        return len(encoding.encode(text))

    async def invoke(self, user_prompt: str, system_prompt_override: Optional[str] = None) -> str:
        """
        调用AI模型并获取响应。

        Args:
            user_prompt (str): 用户的输入提示。
            system_prompt_override (Optional[str]): 可选的，用于覆盖代理默认系统提示的临时提示。

        Returns:
            str: AI模型的响应文本。
        """
        # 如果提供了覆盖提示，则使用它；否则，使用代理的默认系统提示
        active_system_prompt = system_prompt_override if system_prompt_override is not None else self.system_prompt

        messages = [
            {"role": "system", "content": active_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.verbose:
            print(f"\n--- 向 '{self.name}' 发送的提示 ---")
            print(user_prompt)
            print("---------------------------------")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            
            if self.verbose:
                print(f"\n--- 从 '{self.name}' 收到的原始响应 ---")
                print(response)
                print("------------------------------------")

            # 记录token使用量
            if response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens

            content = response.choices[0].message.content
            return content if content else ""

        except Exception as e:
            if self.verbose:
                print(f"代理 '{self.name}' 在调用API时出错: {e}")
            return f"错误: {e}"

    def get_usage_stats(self) -> Dict[str, int]:
        """
        获取此代理的token使用情况。

        Returns:
            Dict[str, int]: 包含提示token、完成token和总token的字典。
        """
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

async def main_test():
    """异步测试函数"""
    from config_loader import load_config, get_agent_settings

    print("正在测试 Agent 类...")
    try:
        # 加载配置
        app_config = load_config()
        generator_settings = get_agent_settings("solution_generator", app_config)

        if not generator_settings.get("api_key"):
            raise ValueError("未找到 API 密钥。请检查你的 .env 文件。")

        # 创建一个Agent实例
        solution_generator = Agent(
            name="solution_generator",
            model=generator_settings["model"],
            api_key=generator_settings["api_key"],
            base_url=generator_settings["base_url"],
            system_prompt="你是一个富有创造力的解决方案架构师。请为给定的问题提供三个不同的、创新的解决方案思路。",
            verbose=True
        )

        # 调用Agent
        problem = "如何提高城市垃圾分类的效率？"
        print(f"正在向 '{solution_generator.name}' 提问: {problem}")
        response_text = await solution_generator.invoke(problem)

        # 打印结果
        print("\nAI的响应:")
        print("="*20)
        print(response_text)
        print("="*20)

        # 打印token使用情况
        usage = solution_generator.get_usage_stats()
        print(f"\nToken 使用情况: {usage}")

    except (ValueError, FileNotFoundError) as e:
        print(f"\n测试失败: {e}")
        print("请确保 config.yml 和 .env 文件已正确配置。")
    except Exception as e:
        print(f"\n发生未知错误: {e}")

if __name__ == '__main__':
    # 这是一个简单的测试，用于验证Agent类是否正常工作
    # 在运行此测试之前，请确保你已经：
    # 1. 创建了 .env 文件并填入了你的 OPENAI_API_KEY。
    # 2. 运行 `pip install -r requirements.txt` 安装了所有依赖。
    asyncio.run(main_test())
