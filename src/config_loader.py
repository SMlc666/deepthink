import os
import yaml
from dotenv import load_dotenv
from typing import List, Dict, Any

def load_config() -> Dict[str, Any]:
    """
    加载 config.yml 文件。
    """
    with open("config.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_environment():
    """
    加载 .env 文件中的环境变量。
    """
    load_dotenv()

def get_agent_settings(agent_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据代理名称从配置中获取其设置，并结合环境变量返回最终配置。

    Args:
        agent_name: 代理的名称 (例如 "solution_generator")。
        config: 从 config.yml 加载的配置字典。

    Returns:
        一个包含 model, api_key, 和 base_url 的字典。
    
    Raises:
        ValueError: 如果在配置中找不到指定的代理。
    """
    agent_config = next((agent for agent in config.get("agents", []) if agent["name"] == agent_name), None)
    
    if not agent_config:
        raise ValueError(f"在 config.yml 中未找到名为 '{agent_name}' 的代理配置。")

    provider = agent_config.get("provider")
    
    if provider:
        # 如果指定了 provider，则从环境变量中查找对应的 KEY 和 URL
        api_key = os.getenv(f"{provider.upper()}_API_KEY")
        base_url = os.getenv(f"{provider.upper()}_BASE_URL")
    else:
        # 否则，使用默认的 OpenAI 环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

    return {
        "model": agent_config["model"],
        "api_key": api_key,
        "base_url": base_url if base_url else None, # 确保 None 而不是空字符串
    }

# 在模块加载时自动加载环境变量
load_environment()

if __name__ == '__main__':
    # 这是一个简单的测试，用于验证加载功能是否正常
    # 在实际运行前，请确保你已经创建了 .env 文件并填入了你的 API KEY
    
    # 1. 加载 .env 文件 (已在模块顶部自动完成)
    # 2. 加载 config.yml
    app_config = load_config()
    print("config.yml 加载成功:")
    print(app_config)
    print("-" * 20)

    # 3. 获取特定代理的配置
    try:
        generator_settings = get_agent_settings("solution_generator", app_config)
        print("solution_generator 配置:")
        print(generator_settings)
        print("-" * 20)

        reviewer_settings = get_agent_settings("final_reviewer", app_config)
        print("final_reviewer 配置:")
        print(reviewer_settings)
        print("-" * 20)
        
    except (ValueError, FileNotFoundError) as e:
        print(f"错误: {e}")
        print("请确保 config.yml 和 .env 文件存在且配置正确。")
