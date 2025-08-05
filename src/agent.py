import openai
import tiktoken
import asyncio
import traceback
from typing import List, Dict, Optional, AsyncIterator, Tuple

from .errors import EmptyApiResponseError

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

    def _handle_api_error(self, e: Exception, attempt: int, max_retries: int) -> Tuple[bool, Optional[str]]:
        """
        集中处理API调用中的各种异常。

        Args:
            e (Exception): 捕获到的异常。
            attempt (int): 当前的尝试次数。
            max_retries (int): 最大重试次数。

        Returns:
            一个元组 (should_retry, error_message)。
            - should_retry (bool): 是否应该重试。
            - error_message (Optional[str]): 如果不应重试，则返回的错误消息。
        """
        if self.verbose:
            print(f"代理 '{self.name}' 在调用API时遇到错误 (尝试 {attempt + 1}/{max_retries}): {e}")

        if isinstance(e, (openai.APIConnectionError, openai.RateLimitError, openai.APITimeoutError, EmptyApiResponseError)) or "RemoteProtocolError" in str(e):
            # 对于可重试的错误
            if attempt < max_retries - 1:
                return True, None  # 应该重试
            else:
                # 达到最大重试次数
                error_message = f"错误: 代理 '{self.name}' 在 {max_retries} 次尝试后调用API失败。最后一次错误: {e}"
                if self.verbose:
                    print(error_message)
                return False, error_message
        elif isinstance(e, openai.APIStatusError):
            # 对于API状态错误，通常不应重试
            if self.verbose:
                print(f"代理 '{self.name}' 收到来自API的错误状态码: {e.status_code}")
                traceback.print_exc()
            return False, f"错误: API返回错误状态 {e.status_code}。响应: {e.response}"
        else:
            # 对于其他未知错误
            if self.verbose:
                print(f"代理 '{self.name}' 在调用API时发生未知错误。")
                traceback.print_exc()
            return False, f"错误: 发生未知错误, 错误: {e}"

    async def invoke(self, user_prompt: str, system_prompt_override: Optional[str] = None) -> AsyncIterator[str]:
        """
        以流式方式调用AI模型并获取响应。

        Args:
            user_prompt (str): 用户的输入提示。
            system_prompt_override (Optional[str]): 可选的，用于覆盖代理默认系统提示的临时提示。

        Yields:
            str: AI模型响应的文本块或最终的错误信息。
        """
        active_system_prompt = system_prompt_override if system_prompt_override is not None else self.system_prompt
        messages = [
            {"role": "system", "content": active_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt_tokens = self._count_tokens(active_system_prompt + user_prompt)
        self.total_prompt_tokens += prompt_tokens

        if self.verbose:
            print(f"\n--- 向 '{self.name}' 发送的提示 (流式) ---")
            print(f"Prompt Tokens: {prompt_tokens}")
            print(user_prompt)
            print("---------------------------------")

        max_retries = 3
        delay = 2.0

        for attempt in range(max_retries):
            try:
                response_stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )
                
                completion_tokens = 0
                content_received = False
                async for chunk in response_stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        content_received = True
                        completion_tokens += self._count_tokens(content)
                        yield content
                
                # 如果流结束了但没有收到任何内容，则抛出异常
                if not content_received:
                    raise EmptyApiResponseError("API成功返回但响应流为空。")

                self.total_completion_tokens += completion_tokens
                if self.verbose:
                    print(f"\n--- '{self.name}' 流式传输完成 ---")
                    print(f"Completion Tokens: {completion_tokens}")
                    print("------------------------------------")
                return  # 成功完成，退出循环

            except Exception as e:
                should_retry, error_message = self._handle_api_error(e, attempt, max_retries)
                if should_retry:
                    await asyncio.sleep(delay)
                    delay *= 2.0
                else:
                    yield error_message
                    return

    async def invoke_non_stream(self, user_prompt: str, system_prompt_override: Optional[str] = None) -> str:
        """
        调用AI模型并获取完整的响应。
        内部使用流式API并聚合结果。

        Args:
            user_prompt (str): 用户的输入提示。
            system_prompt_override (Optional[str]): 可选的，用于覆盖代理默认系统提示的临时提示。

        Returns:
            str: AI模型的完整响应文本。
        """
        full_response = []
        async for chunk in self.invoke(user_prompt, system_prompt_override):
            full_response.append(chunk)
        return "".join(full_response)

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
    from .config_loader import load_config, get_agent_settings

    print("正在测试 Agent 类...")
    try:
        # 加载配置
        app_config = load_config()
        generator_settings = get_agent_settings("solution_generator", app_config)

        if not generator_settings.get("api_key"):
            raise ValueError("未找到 API 密钥。请检查你的 .env 文件。" )

        # 创建一个Agent实例
        solution_generator = Agent(
            name="solution_generator",
            model=generator_settings["model"],
            api_key=generator_settings["api_key"],
            base_url=generator_settings["base_url"],
            system_prompt="你是一个富有创造力的解决方案架构师。请为给定的问题提供三个不同的、创新的解决方案思路。",
            verbose=True
        )

        # --- 测试流式调用 ---
        problem = "如何提高城市垃圾分类的效率？"
        print(f"\n--- 正在向 '{solution_generator.name}' 进行流式提问 ---")
        print(f"问题: {problem}")
        
        print("\nAI的流式响应:")
        print("="*20)
        response_stream = solution_generator.invoke(problem)
        full_response_text = ""
        async for chunk in response_stream:
            print(chunk, end="", flush=True)
            full_response_text += chunk
        print("\n" + "="*20)

        # 打印token使用情况
        usage = solution_generator.get_usage_stats()
        print(f"\nToken 使用情况 (流式): {usage}")

        # --- 测试非流式调用 ---
        print(f"\n--- 正在向 '{solution_generator.name}' 进行非流式提问 ---")
        problem_non_stream = "写一个Python函数，计算斐波那契数列的第n项。"
        print(f"问题: {problem_non_stream}")
        
        response_text_non_stream = await solution_generator.invoke_non_stream(problem_non_stream)

        print("\nAI的非流式响应:")
        print("="*20)
        print(response_text_non_stream)
        print("="*20)
        
        usage_non_stream = solution_generator.get_usage_stats()
        print(f"\nToken 使用情况 (非流式之后): {usage_non_stream}")


    except (ValueError, FileNotFoundError) as e:
        print(f"\n测试失败: {e}")
        print("请确保 config.yml 和 .env 文件已正确配置。")
    except Exception as e:
        import traceback
        print(f"\n发生未知错误: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    # 这是一个简单的测试，用于验证Agent类是否正常工作
    # 在运行此测试之前，请确保你已经：
    # 1. 创建了 .env 文件并填入了你的 OPENAI_API_KEY。
    # 2. 运行 `pip install -r requirements.txt` 安装了所有依赖。
    asyncio.run(main_test())
