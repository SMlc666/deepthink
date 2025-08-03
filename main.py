import asyncio
import argparse
import traceback
from src.orchestrator import Orchestrator
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory

async def main():
    """
    应用程序的主入口点。
    解析命令行参数并启动Orchestrator。
    """
    parser = argparse.ArgumentParser(
        description="使用多AI代理工作流解决复杂问题。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "problem",
        nargs="?",  # 将 problem 设置为可选参数
        type=str,
        default=None,
        help="需要AI解决的问题或任务。如果未提供，将进入交互模式。"
    )
    
    parser.add_argument(
        "--solutions",
        type=int,
        default=3,
        help="希望生成的解决方案数量。 (默认: 3)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="batch",
        choices=["sequential", "batch"],
        help="""选择工作流执行模式:
  - sequential: 串行模式，按顺序处理每个想法（批判 -> 执行）。
  - batch: 批处理模式，并行处理所有任务以提高速度。 (默认)
"""
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="启用详细输出模式，显示更多调试信息。"
    )

    args = parser.parse_args()

    problem = args.problem
    if not problem:
        try:
            # 如果命令行没有提供问题，则进入交互模式
            history = FileHistory('.app_history')
            problem = prompt("请输入您想解决的问题: ", history=history)
            if not problem:
                print("没有输入问题，程序退出。")
                return
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            return

    try:
        orchestrator = Orchestrator(verbose=args.verbose)
        await orchestrator.run(
            problem=problem,
            num_solutions=args.solutions,
            mode=args.mode
        )
    except FileNotFoundError:
        print("\n错误: 'config.yml' 未找到。")
        print("请确保配置文件存在于项目的根目录中。")
    except Exception as e:
        print(f"\n运行过程中发生未知错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 在运行此程序之前，请确保你已经：
    # 1. 创建了 .env 文件并填入了你的 API 密钥。
    # 2. 运行 `pip install -r requirements.txt` 安装了所有依赖。
    #
    # 示例用法:
    # python main.py "我们如何利用AI技术来改善在线教育的个性化学习体验？"
    # python main.py "如何为一家小型咖啡店设计一个忠诚度计划？" --mode sequential --solutions 2
    
    asyncio.run(main())