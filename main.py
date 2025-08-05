import asyncio
import argparse
import traceback
from src.orchestrator import Orchestrator
from src import history_manager 
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

def display_history():
    """Displays the last 10 history entries."""
    history = history_manager.load_history()
    if not history:
        print("没有历史记录。")
        return
    
    print("--- 最近的10条记录 ---")
    for i, entry in enumerate(history[:10]):
        status_icon = "✓" if entry.get('status') == 'completed' else "✗" if entry.get('status') == 'failed' else "…"
        print(f"{i+1}. [{status_icon}] {entry['timestamp'][:19]} - {entry['problem'][:50]}...")
    print("--------------------")

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
        help="""
选择工作流执行模式:
  - sequential: 串行模式，按顺序处理每个想法（批判 -> 执行）。
  - batch: 批处理模式，并行处理所有任务以提高速度。 (默认)
"""
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="启用详细输出模式，显示更多调试信息。"
    )
    
    parser.add_argument(
        "--history",
        action="store_true",
        help="显示最近的10条历史记录并退出。"
    )

    args = parser.parse_args()

    if args.history:
        display_history()
        return

    problem = args.problem
    if not problem:
        try:
            # 如果命令行没有提供问题，则进入交互模式
            history = FileHistory('.app_history')
            session = PromptSession(history=history)
            problem = await session.prompt_async("请输入您想解决的问题: ")
            if not problem:
                print("没有输入问题，程序退出。")
                return
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            return

    # Create history entry
    history_entry = history_manager.add_history_entry(
        problem=problem,
        mode=args.mode,
        solutions=args.solutions,
        source='cli'
    )
    history_id = history_entry['id']

    try:
        orchestrator = Orchestrator(verbose=args.verbose)
        final_review_content = None
        # 由于 run 方法现在是异步生成器，我们需要迭代它来驱动流程
        async for update in orchestrator.run(
            problem=problem,
            num_solutions=args.solutions,
            mode=args.mode,
            history_id=history_id
        ):
            # 为了保留CLI的日志功能，我们可以在这里打印更新
            if args.verbose:
                print(f"[UPDATE] {update.get('title', update.get('event'))}: {update.get('nodeId')}")
            # 最终的评审结果会在最后一步打印
            if update.get('nodeId') == 'final_review' and update.get('event') == 'completed':
                final_review_content = update.get('content')
                print("\n\n--- 最终评审结果 ---")
                print(final_review_content)
        
        # Update history on completion
        history_manager.update_history_entry(history_id, {
            "status": "completed",
            "final_review": final_review_content,
            "usage": orchestrator.get_total_usage(),
            "graph_data": None # CLI runs don't save graph data for now
        })


    except FileNotFoundError:
        print("\n错误: 'config.yml' 未找到。")
        print("请确保配置文件存在于项目的根目录中。")
        history_manager.update_history_entry(history_id, {"status": "failed"})
    except Exception as e:
        print(f"\n运行过程中发生未知错误: {e}")
        traceback.print_exc()
        history_manager.update_history_entry(history_id, {"status": "failed"})

if __name__ == "__main__":
    # 在运行此程序之前，请确保你已经：
    # 1. 创建了 .env 文件并填入了你的 API 密钥。
    # 2. 运行 `pip install -r requirements.txt` 安装了所有依赖。
    #
    # 示例用法:
    # python main.py "我们如何利用AI技术来改善在线教育的个性化学习体验？"
    # python main.py "如何为一家小型咖啡店设计一个忠诚度计划？" --mode sequential --solutions 2
    # python main.py --history
    
    asyncio.run(main())
