import json
import asyncio
import re
from typing import Dict, Any, List, AsyncGenerator, Optional
from .config_loader import load_config, get_agent_settings
from .agent import Agent
from .prompts import get_prompt
from .code_executor import execute_code
from . import tools
from . import history_manager

class Orchestrator:
    """
    工作流编排器，负责管理和执行整个AI协作流程。
    """
    def __init__(self, verbose: bool = False):
        """
        初始化编排器，加载配置并创建代理实例。
        """
        self.verbose = verbose
        if self.verbose:
            print("正在初始化编排器...")
        
        self.config = load_config()
        self.agents: Dict[str, Agent] = self._create_agents()
        self.available_libraries = self._load_available_libraries()
        self.total_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if self.verbose:
            print("编排器初始化完成。")
            self._log(f"可用的库: {self.available_libraries}")

        # 检查搜索功能是否可用
        self.search_tools_available = "searcher" in self.agents and "summarizer" in self.agents
        if self.verbose:
            self._log(f"搜索工具状态: {'可用' if self.search_tools_available else '不可用'}")

    def _log(self, message: str):
        """如果启用详细模式，则打印日志消息。"""
        if self.verbose:
            print(message)

    def _create_agents(self) -> Dict[str, Agent]:
        """
        根据配置文件创建所有代理。
        """
        agents = {}
        for agent_config in self.config.get("agents", []):
            name = agent_config["name"]
            settings = get_agent_settings(name, self.config)
            
            if not settings.get("api_key"):
                self._log(f"警告: 未找到代理 '{name}' 的 API 密钥，将跳过创建。")
                continue

            system_prompt = get_prompt(name)

            # 为 critic agent 加载两个不同的 prompt
            if name == 'critic':
                agents['execution_critic'] = Agent(
                    name='execution_critic',
                    model=settings["model"],
                    api_key=settings["api_key"],
                    base_url=settings["base_url"],
                    system_prompt=get_prompt('execution_critic'),
                    verbose=self.verbose
                )
                if self.verbose:
                    self._log(f"成功创建代理: 'execution_critic'")

            agents[name] = Agent(
                name=name,
                model=settings["model"],
                api_key=settings["api_key"],
                base_url=settings["base_url"],
                system_prompt=system_prompt,
                verbose=self.verbose
            )
            if self.verbose:
                self._log(f"成功创建代理: '{name}'")
        return agents

    def _load_available_libraries(self) -> List[str]:
        """从 requirements.txt 加载可用库的列表。"""
        try:
            with open("requirements.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
                libraries = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # 移除版本说明等，只保留库名
                        lib_name = re.split(r'[=<>~]', line)[0].strip()
                        libraries.append(lib_name)
                return libraries
        except FileNotFoundError:
            self._log("警告: requirements.txt 未找到，无法加载可用库列表。")
            return []

    def get_total_usage(self) -> Dict[str, int]:
        """返回运行的总token使用情况。"""
        return self.total_usage

    async def run(self, problem: str, num_solutions: int = 3, mode: str = "sequential", history_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行完整的工作流程，并以异步生成器的方式yield出每一步的更新。
        """
        self._log(f"\n{'='*20} 开始处理新问题 ({mode} 模式) {'='*20}")
        self._log(f"问题: {problem}")

        # 根节点
        root_node_id = "root"
        yield {
            "event": "progress", "nodeId": root_node_id, "title": "开始处理",
            "content": f"问题: {problem}\n模式: {mode}\n方案数: {num_solutions}"
        }

        # 阶段 0: 问题路由
        router_node_id = "router"
        yield {
            "event": "progress", "nodeId": router_node_id, "parentId": root_node_id, "title": "问题路由",
            "content": "正在分析问题类型..."
        }
        router = self.agents.get("router")
        if not router:
            self._log("错误: 未找到 'router' 代理。将默认使用工程工作流。 ")
            problem_type = "ENGINEER"
        else:
            problem_type = (await router.invoke_non_stream(problem)).strip().upper()
        
        yield {
            "event": "completed", "nodeId": router_node_id, "parentId": root_node_id, "title": f"问题路由 ({problem_type})",
            "content": f"问题类型被判断为: {problem_type}"
        }

        # 根据问题类型选择工作流
        if problem_type == "MATHEMATICIAN":
            final_results_generator = self._run_mathematical_workflow(problem, num_solutions, router_node_id)
        elif problem_type == "GENERAL":
            final_content = ""
            # 对于通用问题，我们直接运行通用工作流
            async for event in self._run_deep_general_workflow(problem, router_node_id):
                yield event
                if event.get("nodeId") == "general_editor" and event.get("event") == "completed":
                    final_content = event.get("content")

            # 通用流程结束后，直接结束整个过程
            self._summarize_usage()
            if history_id:
                history = history_manager.load_history()
                for entry in history:
                    if entry.get("id") == history_id:
                        entry["status"] = "completed"
                        entry["final_review"] = final_content
                        entry["usage"] = self.get_total_usage()
                        # The caller (app.py) will add the graph_data
                        break
                history_manager.save_history(history)

            self._log(f"\n{'='*20} 流程处理结束 {'='*20}")
            yield {"event": "completed", "nodeId": root_node_id, "content": f"流程处理结束.\n\nToken使用情况:\n{json.dumps(self.total_usage, indent=2)}"}
            return
        else: # 默认为 ENGINEER
            final_results_generator = self._run_engineering_workflow(problem, num_solutions, mode, router_node_id)

        final_results = []
        async for result in final_results_generator:
            if "event" in result:
                yield result
            else:
                final_results.append(result)


        if not final_results:
            self._log("工作流未能产出最终结果，流程终止。")
            if history_id:
                history_manager.update_history_entry(history_id, {"status": "failed"})
            yield {"event": "failed", "nodeId": root_node_id, "content": "工作流未能产出最终结果"}
            return

        # 最终评审
        review_node_id = "final_review"
        yield {
            "event": "progress", "nodeId": review_node_id, "parentId": root_node_id, "title": "最终评审",
            "content": "正在对所有方案进行最终评审..."
        }
        
        final_review_content = ""
        async for event in self._final_review(problem, final_results, review_node_id):
            if event.get("event") == "chunk":
                final_review_content += event.get("content", "")
            yield event

        yield {
            "event": "completed", "nodeId": review_node_id, "parentId": root_node_id, "title": "最终评审完成",
            "content": final_review_content
        }

        self._summarize_usage()
        if history_id:
            history = history_manager.load_history()
            for entry in history:
                if entry.get("id") == history_id:
                    entry["status"] = "completed"
                    entry["final_review"] = final_review_content
                    entry["usage"] = self.get_total_usage()
                    # The caller (app.py) will add the graph_data
                    break
            history_manager.save_history(history)

        self._log(f"\n{'='*20} 流程处理结束 {'='*20}")
        yield {"event": "completed", "nodeId": root_node_id, "content": f"流程处理结束。\n\nToken使用情况:\n{json.dumps(self.total_usage, indent=2)}"}

    async def _run_generators_in_parallel(self, generators: List):
        queue = asyncio.Queue()
        END_OF_GENERATOR = object()

        async def producer(gen):
            try:
                async for item in gen:
                    await queue.put(item)
            finally:
                await queue.put(END_OF_GENERATOR)

        producer_tasks = [asyncio.create_task(producer(gen)) for gen in generators]
        
        finished_producers = 0
        while finished_producers < len(producer_tasks):
            item = await queue.get()
            if item is END_OF_GENERATOR:
                finished_producers += 1
            else:
                yield item

    async def _run_engineering_workflow(self, problem: str, num_solutions: int, mode: str, parent_node_id: str):
        self._log("\n--- 工程工作流执行 ---")

        ideas_node_id = "generate_ideas"
        yield {
            "event": "progress", "nodeId": ideas_node_id, "parentId": parent_node_id, "title": "生成思路",
            "content": f"正在生成 {num_solutions} 个解决方案思路..."
        }
        ideas = await self._generate_ideas(problem, num_solutions)
        if not ideas:
            yield {"event": "failed", "nodeId": ideas_node_id, "content": "未能生成任何思路"}
            return

        try:
            json_str = ""
            match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', ideas)
            if match:
                json_str = match.group(1)
            else:
                match = re.search(r'(\{[\s\S]*\})', ideas)
                if match:
                    json_str = match.group(1)

            if not json_str:
                raise json.JSONDecodeError("在AI响应中找不到JSON对象或代码块。", ideas, 0)

            ideas_json = json.loads(json_str)
            solutions = ideas_json.get("solutions", [])
            yield {
                "event": "completed", "nodeId": ideas_node_id, "title": f"生成了 {len(solutions)} 个思路",
                "content": ideas
            }
        except json.JSONDecodeError:
            yield {"event": "failed", "nodeId": ideas_node_id, "content": f"未能返回有效的JSON格式:\n{ideas}"}
            return

        if not solutions:
            return

        if mode == "batch":
            solution_generators = [self._process_solution_in_parallel(solution, i, ideas_node_id) for i, solution in enumerate(solutions)]
            async for item in self._run_generators_in_parallel(solution_generators):
                yield item
        else: 
            final_results = []
            for i, solution in enumerate(solutions):
                async for item in self._run_sequential_workflow_for_solution(solution, i, ideas_node_id):
                    if "event" not in item:
                         final_results.append(item)
                    else:
                        yield item
            for res in final_results:
                yield res


    async def _run_sequential_workflow_for_solution(self, solution: Dict[str, Any], index: int, parent_node_id: str):
        solution_node_id = f"solution_{index}"
        title = solution.get('title', f'思路 {index+1}')
        yield {
            "event": "progress", "nodeId": solution_node_id, "parentId": parent_node_id, "title": title,
            "content": f"开始处理: {title}"
        }

        critique_node_id = f"critique_{index}"
        yield {
            "event": "progress", "nodeId": critique_node_id, "parentId": solution_node_id, "title": f"批判: {title}",
            "content": "正在进行初始思路批判..."
        }
        
        initial_criticism = ""
        critic = self.agents.get("critic")
        if critic:
            prompt = f"请批判以下解决方案思路：\n\n标题: {solution.get('title')}\n\n描述: {solution.get('description')}\n\n步骤: {solution.get('steps')}"
            async for chunk in critic.invoke(prompt):
                yield {"event": "chunk", "nodeId": critique_node_id, "content": chunk}
                initial_criticism += chunk
        else:
            initial_criticism = "错误: 未找到 'critic' 代理。"

        yield {
            "event": "completed", "nodeId": critique_node_id, "title": f"批判完成: {title}",
            "content": initial_criticism
        }

        execution_log = ""
        async for update in self._execute_and_critique_loop(solution, initial_criticism, solution_node_id, index):
            if "event" in update:
                yield update
            else:
                execution_log = update

        result = {
            "solution": solution,
            "criticism": initial_criticism,
            "execution_log": execution_log
        }
        yield {
            "event": "completed", "nodeId": solution_node_id, "title": f"处理完成: {title}",
            "content": json.dumps(result, indent=2, ensure_ascii=False)
        }
        yield result

    async def _process_solution_in_parallel(self, solution: Dict[str, Any], index: int, parent_node_id: str):
        solution_node_id = f"solution_{index}"
        title = solution.get('title', f'思路 {index+1}')
        yield {
            "event": "progress", "nodeId": solution_node_id, "parentId": parent_node_id, "title": title,
            "content": f"开始处理: {title}"
        }

        critique_node_id = f"critique_{index}"
        yield { "event": "progress", "nodeId": critique_node_id, "parentId": solution_node_id, "title": f"批判: {title}" }
        
        initial_criticism = ""
        critic = self.agents.get("critic")
        if critic:
            prompt = f"请批判以下解决方案思路：\n\n标题: {solution.get('title')}\n\n描述: {solution.get('description')}\n\n步骤: {solution.get('steps')}"
            async for chunk in critic.invoke(prompt):
                yield {"event": "chunk", "nodeId": critique_node_id, "content": chunk}
                initial_criticism += chunk
        else:
            initial_criticism = "错误: 未找到 'critic' 代理。"
        
        yield { "event": "completed", "nodeId": critique_node_id, "content": initial_criticism }
        
        execution_log = ""
        async for update in self._execute_and_critique_loop(solution, initial_criticism, solution_node_id, index):
            if "event" in update:
                yield update
            else:
                execution_log = update

        result = {
            "solution": solution,
            "criticism": initial_criticism,
            "execution_log": execution_log
        }
        yield {
            "event": "completed", "nodeId": solution_node_id, "title": f"处理完成: {title}",
            "content": json.dumps(result, indent=2, ensure_ascii=False)
        }
        yield result

    async def _generate_ideas(self, problem: str, num_solutions: int) -> str:
        self._log("\n--- 阶段 1: 生成解决方案思路 ---")
        generator = self.agents.get("solution_generator")
        if not generator:
            self._log("错误: 未找到 'solution_generator' 代理。")
            return ""

        prompt_template = get_prompt("solution_generator")
        formatted_prompt = prompt_template.format(num_solutions=num_solutions)
        
        response = await generator.invoke_non_stream(problem, system_prompt_override=formatted_prompt)
        self._log(f"'{generator.name}' 已生成思路。")
        return response

    async def _critique_idea(self, solution: Dict[str, Any], execution_log: str = "") -> str:
        self._log("  - 正在进行批判性分析...")
        
        if execution_log:
            critic = self.agents.get("execution_critic")
            prompt_template_name = "execution_critic"
        else:
            critic = self.agents.get("critic")
            prompt_template_name = "critic"

        if not critic: return f"错误: 未找到 '{prompt_template_name}' 代理。"
        
        if execution_log:
             prompt = f"任务: {solution.get('title')}\n\n执行日志:\n{execution_log}"
        else:
             prompt = f"请批判以下解决方案思路：\n\n标题: {solution.get('title')}\n\n描述: {solution.get('description')}\n\n步骤: {solution.get('steps')}"

        response = await critic.invoke_non_stream(prompt)
        self._log("  - 批判完成。")
        return response

    async def _execute_and_critique_loop(self, solution: Dict[str, Any], initial_criticism: str, parent_node_id: str, index: int):
        self._log("  - 进入执行与迭代批判循环...")
        
        loop_node_id = f"loop_{index}"
        yield {
            "event": "progress", "nodeId": loop_node_id, "parentId": parent_node_id, "title": "执行 & 迭代",
            "content": "开始执行与迭代循环..."
        }

        executor = self.agents.get("executor")
        if not executor:
            yield {"event": "failed", "nodeId": loop_node_id, "content": "错误: 未找到 'executor' 代理。"}
            yield "错误: 未找到 'executor' 代理。"
            return

        available_libs_str = ", ".join(self.available_libraries)
        search_instruction = "如果需要外部知识，请使用搜索工具。否则，请编写代码。" if self.search_tools_available else "搜索工具当前不可用。"

        current_prompt = (
            f"任务：基于以下解决方案思路和批判意见，来完成任务。\n\n"
            f"**你可以使用的Python库**: {available_libs_str}\n\n"
            f"--- 原始思路 ---\n标题: {solution.get('title')}\n描述: {solution.get('description')}\n\n"
            f"--- 初始批判意见 ---\n{initial_criticism}\n\n"
            f"请开始你的第一步。分析问题，制定计划。{search_instruction}"
        )

        conversation_history = [f"Orchestrator: {current_prompt}"]
        max_iterations = 8

        for i in range(max_iterations):
            iter_node_id = f"{loop_node_id}_iter_{i}"
            yield {
                "event": "progress", "nodeId": iter_node_id, "parentId": loop_node_id, "title": f"迭代 {i+1}",
                "content": "执行者正在思考..."
            }
            
            executor_response = ""
            async for chunk in executor.invoke(current_prompt):
                yield {"event": "chunk", "nodeId": iter_node_id, "content": chunk}
                executor_response += chunk

            conversation_history.append(f"Executor:\n{executor_response}")
            
            tool_match = re.search(r'\[TOOL_REQUEST\]\s*(\{[\s\S]*\})\s*', executor_response, re.DOTALL)
            code_match = re.search(r'```python\n([\s\S]*?)\n```', executor_response, re.DOTALL)
            feedback_for_next_round = ""

            if tool_match:
                pass 
            elif code_match:
                code_to_execute = code_match.group(1)
                execution_result = execute_code(code_to_execute)
                execution_result_str = f"[EXECUTION_RESULT]\nSTDOUT:\n{execution_result['stdout']}\n\nSTDERR:\n{execution_result['stderr']}\n"
                conversation_history.append(f"CodeExecutor:\n{execution_result_str}")
                
                execution_criticism = await self._critique_idea(solution, execution_result_str)
                conversation_history.append(f"ExecutionCritic:\n{execution_criticism}")
                
                yield {
                    "event": "completed", "nodeId": iter_node_id, "title": f"迭代 {i+1} 完成",
                    "content": f"执行者响应:\n{executor_response}\n\n执行结果:\n{execution_result_str}\n\n批判意见:\n{execution_criticism}"
                }

                if "[ACCEPTABLE]" in execution_criticism:
                    break
                feedback_for_next_round = f"{execution_result_str}\n这是对你执行结果的批判意见:\n{execution_criticism}\n\n"
            else:
                feedback_for_next_round = "你的回复中既没有代码块也没有工具调用。请根据任务，决定是使用工具还是编写代码。\n"
                yield {
                    "event": "failed", "nodeId": iter_node_id, "title": f"迭代 {i+1} 失败",
                    "content": f"执行者响应:\n{executor_response}\n\n系统反馈:\n{feedback_for_next_round}"
                }

            if "[COMPLETE]" in executor_response:
                break
            if i == max_iterations - 1:
                conversation_history.append("Orchestrator: 达到最大迭代次数，流程终止。")
                break

            current_prompt = (
                f"这是你之前的尝试:\n{executor_response}\n\n"
                f"这是来自系统或工具的反馈:\n{feedback_for_next_round}\n\n"
                "请分析反馈，修正你的计划，然后继续下一步。"
            )

        final_log = "\n\n---".join(conversation_history)
        yield {"event": "completed", "nodeId": loop_node_id, "content": final_log}
        yield final_log

    async def _run_search_workflow(self, query_context: str, parent_node_id: str):
        self._log("\n--- 搜索工作流启动 ---")
        
        searcher = self.agents.get("searcher")
        summarizer = self.agents.get("summarizer")
        if not searcher or not summarizer:
            yield {"event": "failed", "nodeId": parent_node_id, "content": "错误: 未找到 'searcher' 或 'summarizer' 代理。"}
            return

        searcher_node_id = f"{parent_node_id}_searcher"
        yield {"event": "progress", "nodeId": searcher_node_id, "parentId": parent_node_id, "title": "生成搜索查询", "content": query_context}
        searcher_response = await searcher.invoke_non_stream(query_context)
        
        try:
            json_str = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', searcher_response, re.DOTALL)
            search_request = json.loads(json_str.group(1) if json_str else searcher_response)
            search_query = search_request.get("query")
            if not search_query:
                raise ValueError("Searcher未能生成有效的搜索查询。 ")
            yield {"event": "completed", "nodeId": searcher_node_id, "content": f"生成查询: {search_query}"}
        except (json.JSONDecodeError, ValueError) as e:
            err_msg = f"错误: Searcher返回无效: {e}\n响应: {searcher_response}"
            yield {"event": "failed", "nodeId": searcher_node_id, "content": err_msg}
            return

        bing_node_id = f"{parent_node_id}_bing"
        yield {"event": "progress", "nodeId": bing_node_id, "parentId": parent_node_id, "title": "执行网页搜索", "content": f"正在搜索: {search_query}"}
        search_results = tools.bing_search(search_query, num_results=10)
        if not search_results or "error" in search_results[0]:
            err_msg = f"错误: 必应搜索失败: {search_results}"
            yield {"event": "failed", "nodeId": bing_node_id, "content": err_msg}
            return
        yield {"event": "completed", "nodeId": bing_node_id, "content": f"找到 {len(search_results)} 个结果。"}

        fetch_node_id = f"{parent_node_id}_fetch"
        yield {"event": "progress", "nodeId": fetch_node_id, "parentId": parent_node_id, "title": "抓取网页内容", "content": "正在并行抓取..."}
        fetch_tasks = [tools.fetch_webpage_content(res["link"]) for res in search_results if res.get("link")]
        web_contents = await asyncio.gather(*fetch_tasks)
        combined_content = "\n\n---\n--- 网页分割线 ---\n\n".join(web_contents)
        yield {"event": "completed", "nodeId": fetch_node_id, "content": f"抓取完成，总长度: {len(combined_content)} 字符。"}

        summarizer_node_id = f"{parent_node_id}_summarizer"
        yield {"event": "progress", "nodeId": summarizer_node_id, "parentId": parent_node_id, "title": "总结内容", "content": "正在总结抓取到的内容..."}
        summarizer_prompt = (
            f"**原始查询上下文:**\n{query_context}\n\n"
            f"**以下是为你抓取的网页内容，请根据原始查询上下文进行总结:**\n\n{combined_content}"
        )
        
        summary = ""
        async for chunk in summarizer.invoke(summarizer_prompt):
            yield {"event": "chunk", "nodeId": summarizer_node_id, "content": chunk}
            summary += chunk
        
        yield {"event": "completed", "nodeId": summarizer_node_id, "content": summary}


    async def _run_mathematical_workflow(self, problem: str, num_strategies: int, parent_node_id: str):
        self._log("\n--- 数学工作流执行 ---")

        strategies_node_id = "generate_math_strategies"
        yield {
            "event": "progress", "nodeId": strategies_node_id, "parentId": parent_node_id, "title": "生成数学策略",
            "content": f"正在生成 {num_strategies} 个解题策略..."
        }

        validated_strategies = []
        rejection_feedback = []
        max_retries = 2

        for attempt in range(max_retries + 1):
            if attempt > 0:
                self._log(f"\n--- 策略迭代尝试 {attempt}/{max_retries} ---")
                yield {
                    "event": "progress", "nodeId": f"{strategies_node_id}_retry_{attempt}", "parentId": strategies_node_id,
                    "title": f"策略迭代 ({attempt}/{max_retries})",
                    "content": f"所有初步策略均被否决。正在基于批判意见重新生成策略...\n\n否决原因:\n" + "\n".join(rejection_feedback)
                }
            
            strategies_text = await self._generate_math_strategies(problem, num_strategies, rejection_feedback)
            if not strategies_text:
                yield {"event": "failed", "nodeId": strategies_node_id, "content": f"在尝试 {attempt} 次后，未能生成任何策略。"}
                return

            try:
                json_str = ""
                match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', strategies_text)
                if match:
                    json_str = match.group(1)
                else:
                    match = re.search(r'(\{[\s\S]*\})', strategies_text)
                    if match:
                        json_str = match.group(1)

                if not json_str:
                    raise json.JSONDecodeError("在AI响应中找不到JSON对象或代码块。", strategies_text, 0)

                strategies_json = json.loads(json_str)
                strategies = strategies_json.get("strategies", [])
                yield {
                    "event": "completed", "nodeId": f"{strategies_node_id}_attempt_{attempt}", "parentId": strategies_node_id,
                    "title": f"第 {attempt+1} 轮生成了 {len(strategies)} 个策略",
                    "content": strategies_text
                }
            except json.JSONDecodeError:
                yield {"event": "failed", "nodeId": f"{strategies_node_id}_attempt_{attempt}", "content": f"未能返回有效的JSON格式:\n{strategies_text}"}
                continue

            if not strategies:
                self._log("未能从策略生成器响应中解析出任何策略。")
                continue

            skeptic = self.agents.get("skeptic")
            if not skeptic:
                self._log("警告: 未找到 'skeptic' 代理。将跳过策略审查阶段。")
                validated_strategies = strategies
                break 

            current_rejections = []
            vetting_node_id = f"strategy_vetting_attempt_{attempt}"
            yield {
                "event": "progress", "nodeId": vetting_node_id, "parentId": strategies_node_id, "title": f"第 {attempt+1} 轮策略审查",
                "content": f"怀疑论者正在审查 {len(strategies)} 个策略的基本假设..."
            }
            
            for i, strategy in enumerate(strategies):
                strategy_title = strategy.get('title', f'策略 {i+1}')
                vetting_item_node_id = f"{vetting_node_id}_{i}"
                yield { "event": "progress", "nodeId": vetting_item_node_id, "parentId": vetting_node_id, "title": f"审查: {strategy_title}", "content": "正在评估该策略的可行性..."}

                skeptic_prompt = (f"问题: {problem}\n\n策略: {strategy_title}\n描述: {strategy.get('description')}\n\n你的任务是：评估这个策略的可行性，并提供建设性的反馈。")
                
                skeptic_critique = await skeptic.invoke_non_stream(skeptic_prompt)
                
                # 将批判意见附加到策略上，供后续的 Mathematician 参考
                strategy['critique'] = skeptic_critique
                validated_strategies.append(strategy)
                
                self._log(f"  - 策略 '{strategy_title}' 已审查。")
                yield { "event": "completed", "nodeId": vetting_item_node_id, "title": f"策略审查完成: {strategy_title}", "content": skeptic_critique }
            
            yield { "event": "completed", "nodeId": vetting_node_id, "title": f"第 {attempt+1} 轮审查完成", "content": f"所有 {len(strategies)} 个策略都已附带批判意见进入下一阶段。" }

            # 因为我们现在总是接受所有策略（附带批判），所以直接跳出重试循环
            if strategies:
                self._log("所有策略均已审查，退出迭代循环。")
                break

        if not validated_strategies:
            self._log("所有策略在所有尝试中均被否决，流程终止。")
            return

        self._log(f"\n--- 将并行处理 {len(validated_strategies)} 个最终通过审查的策略 ---")
        strategy_generators = [self._process_strategy_in_parallel(problem, strategy, i, strategies_node_id) for i, strategy in enumerate(validated_strategies)]
        
        async for item in self._run_generators_in_parallel(strategy_generators):
            yield item

    async def _process_strategy_in_parallel(self, problem: str, strategy: Dict[str, Any], index: int, parent_node_id: str):
        strategy_node_id = f"math_strategy_{index}"
        title = strategy.get('title', f'策略 {index+1}')
        yield {
            "event": "progress", "nodeId": strategy_node_id, "parentId": parent_node_id, "title": title,
            "content": f"开始处理策略: {title}"
        }
        
        execution_log = ""
        async for update in self._execute_and_critique_loop_for_math(problem, strategy, strategy_node_id, index):
            if "event" in update:
                yield update
            else:
                execution_log = update

        result = {
            "solution": strategy,
            "criticism": "", 
            "execution_log": execution_log
        }
        yield {
            "event": "completed", "nodeId": strategy_node_id, "title": title,
            "content": json.dumps(result, indent=2, ensure_ascii=False)
        }
        yield result

    async def _generate_math_strategies(self, problem: str, num_strategies: int, rejection_feedback: List[str] = None) -> str:
        if self.verbose:
            print("\n--- 阶段 1: 生成数学解题策略 ---")
        generator = self.agents.get("math_strategy_generator")
        if not generator:
            print("错误: 未找到 'math_strategy_generator' 代理。")
            return ""

        prompt_template = get_prompt("math_strategy_generator")
        
        if rejection_feedback:
            feedback_str = "\n".join(rejection_feedback)
            system_prompt_override = (
                f"你之前的策略因为以下原因被否决了：\n"
                f"--- 否决原因 ---\n{feedback_str}\n--- 结束 ---\n\n"
                f"请仔细分析上述批判意见中的根本性缺陷，并生成 {num_strategies} 个【全新的】、【完全不同的】、能够规避这些特定错误的解题策略。"
                f"确保你的新策略不会再犯同样的逻辑错误。\n\n"
                f"请仍然使用原始的JSON格式输出。"
            )
            user_prompt = problem
        else:
            system_prompt_override = prompt_template.format(num_strategies=num_strategies)
            user_prompt = problem
        
        response = await generator.invoke_non_stream(user_prompt, system_prompt_override=system_prompt_override)

        if self.verbose:
            print(f"'{generator.name}' 已生成策略。")
        return response

    async def _execute_and_critique_loop_for_math(self, problem: str, strategy: Dict[str, Any], parent_node_id: str, index: int):
        self._log(f"  - 进入数学执行与批判循环 for strategy: {strategy.get('title')}")

        loop_node_id = f"math_loop_{index}"
        yield {
            "event": "progress", "nodeId": loop_node_id, "parentId": parent_node_id, "title": "执行 & 迭代",
            "content": "启动“探索者-证明者”工作流..."
        }

        # --- 步骤 1: 数学探路者 (MathExplorer) 进行探索 ---
        explorer_node_id = f"{loop_node_id}_explorer"
        yield {
            "event": "progress", "nodeId": explorer_node_id, "parentId": loop_node_id, "title": "阶段1: 探路者探索",
            "content": "数学探路者正在通过研究特例和编写代码来寻找模式与猜想..."
        }

        explorer = self.agents.get("math_explorer")
        if not explorer:
            err_msg = "错误: 未找到 'math_explorer' 代理。"
            yield {"event": "failed", "nodeId": explorer_node_id, "content": err_msg}
            yield err_msg
            return

        explorer_prompt = (
            f"**原始问题**: {problem}\n\n"
            f"**当前策略**: {strategy.get('title', 'N/A')}\n"
            f"**策略描述**: {strategy.get('description', 'N/A')}\n\n"
            f"**你的任务**: 基于此策略，开始你的探索。研究特例，编写代码进行搜索，并提出你的观察和猜想。"
        )
        
        exploration_result = await explorer.invoke_non_stream(explorer_prompt)
        
        # 执行探路者生成的代码
        explorer_code_match = re.search(r'```python\n([\s\S]*?)\n```', exploration_result, re.DOTALL)
        execution_result_str = "[EXPLORER_EXECUTION_RESULT]\n探路者没有提供可执行代码。\n"
        if explorer_code_match:
            code_to_execute = explorer_code_match.group(1)
            execution_result = execute_code(code_to_execute)
            execution_result_str = f"[EXPLORER_EXECUTION_RESULT]\nSTDOUT:\n{execution_result['stdout']}\n\nSTDERR:\n{execution_result['stderr']}\n"

        full_exploration_output = f"{exploration_result}\n\n{execution_result_str}"

        yield {
            "event": "completed", "nodeId": explorer_node_id, "title": "阶段1: 探路者探索完成",
            "content": full_exploration_output
        }

        # --- 步骤 2: 数学家 (Mathematician) 进行证明 ---
        mathematician_node_id = f"{loop_node_id}_mathematician"
        yield {
            "event": "progress", "nodeId": mathematician_node_id, "parentId": loop_node_id, "title": "阶段2: 数学家证明",
            "content": "数学家正在基于探路者的发现，进行形式化证明或构造..."
        }

        mathematician = self.agents.get("mathematician")
        math_critic = self.agents.get("math_critic")
        if not mathematician or not math_critic:
            err_msg = "错误: 未找到 'mathematician' 或 'math_critic' 代理。"
            yield {"event": "failed", "nodeId": loop_node_id, "content": err_msg}
            yield err_msg
            return

        initial_critique = strategy.get('critique', '无')

        current_prompt = (
            f"**原始问题**: {problem}\n\n"
            f"**当前策略**: {strategy.get('title', 'N/A')}\n"
            f"**来自怀疑论者的初步评估**: \n---\n{initial_critique}\n---\n\n"
            f"**来自探路者的探索结果 (关键输入)**: \n---\n{full_exploration_output}\n---\n\n"
            f"**你的任务**: 基于上述所有信息，特别是探路者的发现，开始你的形式化证明或构造。将探路者的猜想转化为严谨的、普适的数学语言。"
        )

        conversation_history = [f"Orchestrator: {current_prompt}"]
        max_iterations = 10 # 数学家的迭代次数可以适当减少

        for i in range(max_iterations):
            iter_node_id = f"{mathematician_node_id}_iter_{i}"
            yield {
                "event": "progress", "nodeId": iter_node_id, "parentId": mathematician_node_id, "title": f"数学家迭代 {i+1}",
                "content": "数学家正在进行严谨推理..."
            }

            mathematician_response = ""
            async for chunk in mathematician.invoke(current_prompt):
                yield {"event": "chunk", "nodeId": iter_node_id, "content": chunk}
                mathematician_response += chunk
            
            conversation_history.append(f"Mathematician:\n{mathematician_response}")

            if not mathematician_response or not mathematician_response.strip() or "错误: Connection error" in mathematician_response:
                continue

            code_match = re.search(r'```python\n([\s\S]*?)\n```', mathematician_response, re.DOTALL)
            criticism = ""
            execution_result_str = ""

            if code_match:
                code_to_execute = code_match.group(1)
                execution_result = execute_code(code_to_execute)
                execution_result_str = f"[MATHEMATICIAN_EXECUTION_RESULT]\nSTDOUT:\n{execution_result['stdout']}\n\nSTDERR:\n{execution_result['stderr']}\n"
                conversation_history.append(f"CodeExecutor:\n{execution_result_str}")
                critic_prompt = f"这是数学家的工作（包括推理和代码）及其代码执行结果，请审查：\n\n{mathematician_response}\n\n{execution_result_str}"
            else:
                critic_prompt = f"这是数学家的推理步骤，请审查其逻辑的严谨性和正确性：\n\n{mathematician_response}"

            criticism = await math_critic.invoke_non_stream(critic_prompt)
            conversation_history.append(f"MathCritic:\n{criticism}")

            yield {
                "event": "completed", "nodeId": iter_node_id, "title": f"数学家迭代 {i+1} 完成",
                "content": f"数学家响应:\n{mathematician_response}\n\n执行结果:\n{execution_result_str}\n\n批判意见:\n{criticism}"
            }

            if "[ACCEPTABLE]" in criticism and "[COMPLETE]" in mathematician_response:
                break
            
            if i == max_iterations - 1:
                conversation_history.append("Orchestrator: 达到最大迭代次数，流程终止。")
                break

            feedback_for_next_round = (
                f"{execution_result_str}\n"
                f"这是对你工作的批判意见:\n{criticism}\n\n"
            )
            current_prompt = (
                f"你当前正在解决的原始问题是：\n"
                f"--- 问题开始 ---\n{problem}\n--- 问题结束 ---\n\n"
                f"你当前遵循的策略是: **{strategy.get('title')}**\n\n"
                f"你上一步的尝试:\n{mathematician_response}\n\n"
                f"系统对你上一轮尝试的反馈是:\n{feedback_for_next_round}\n\n"
                f"**重要指示**: 请严格围绕上述【原始问题】和【当前策略】，分析反馈并修正你的步骤。如果当前策略被证明是错误的，请在你的回应中明确指出该策略失败，然后以 `[COMPLETE]` 信号结束本次尝试。"
            )

        final_log = "\n\n---".join(conversation_history)
        yield {"event": "completed", "nodeId": loop_node_id, "content": final_log}
        yield final_log

    async def _final_review(self, problem: str, results: List[Dict[str, Any]], node_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        if self.verbose:
            print("\n--- 阶段 4: 最终评审 ---")
        reviewer = self.agents.get("final_reviewer")
        if not reviewer:
            yield {"event": "failed", "nodeId": node_id, "content": "错误: 未找到 'final_reviewer' 代理。"}
            return

        context = f"原始问题: {problem}\n\n"
        for i, result in enumerate(results):
            context += f"--- 方案 {i+1}: {result['solution'].get('title')} ---\n"
            context += f"思路描述: {result['solution'].get('description')}\n"
            context += f"批判意见: {result['criticism']}\n"
            context += f"执行日志: \n---\n{result['execution_log']}\n---\n\n"
            
        prompt = f"请基于以下所有信息，进行最终的、全面的评审，为每个方案打分，并推荐最佳方案。\n\n{context}"
        
        async for chunk in reviewer.invoke(prompt):
            yield {"event": "chunk", "nodeId": node_id, "content": chunk}
        
        if self.verbose:
            print("  - 最终评审完成。")

    def _summarize_usage(self):
        """
        汇总并打印所有代理的token使用情况。
        """
        print("\n--- Token 使用情况汇总 ---")
        total_usage = { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
        for agent in self.agents.values():
            usage = agent.get_usage_stats()
            total_usage["prompt_tokens"] += usage["prompt_tokens"]
            total_usage["completion_tokens"] += usage["completion_tokens"]
            total_usage["total_tokens"] += usage["total_tokens"]
            print(f"代理 '{agent.name}': {usage}")
        
        self.total_usage = total_usage
        print(f"\n总计: {self.total_usage}")


    async def _run_deep_general_workflow(self, problem: str, parent_node_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        为通用类型问题执行一个“思考-扩展-综合”的深度工作流。
        """
        self._log("\n--- 深度通用工作流执行 ---")
        
        # --- 步骤 1: 创意构思师 -> 提出多个角度 ---
        strategist_node_id = "general_strategist"
        yield {
            "event": "progress", "nodeId": strategist_node_id, "parentId": parent_node_id, "title": "构思",
            "content": "创意构思师正在将问题分解为多个分析角度..."
        }
        
        strategist = self.agents.get("creative_strategist")
        if not strategist:
            yield {"event": "failed", "nodeId": strategist_node_id, "content": "错误: 未找到 'creative_strategist' 代理。"}
            return

        perspectives_text = await strategist.invoke_non_stream(problem)
        
        try:
            json_str = ""
            match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', perspectives_text)
            if match:
                json_str = match.group(1)
            else:
                json_str = perspectives_text
            
            perspectives_json = json.loads(json_str)
            perspectives = perspectives_json.get("perspectives", [])
            
            if not perspectives:
                raise ValueError("未能从创意构思师的响应中解析出任何角度。")

            yield {
                "event": "completed", "nodeId": strategist_node_id, "title": f"构思完成 ({len(perspectives)}个角度)",
                "content": perspectives_text
            }
        except (json.JSONDecodeError, ValueError) as e:
            yield {"event": "failed", "nodeId": strategist_node_id, "content": f"处理构思师响应失败: {e}\n原始响应:\n{perspectives_text}"}
            return

        # --- 步骤 2: 内容生成器 -> 并行扩展每个角度 ---
        expansion_node_id = "general_expansion"
        yield {
            "event": "progress", "nodeId": expansion_node_id, "parentId": parent_node_id, "title": "扩展",
            "content": f"内容生成器正在并行扩展 {len(perspectives)} 个角度..."
        }
        
        generator = self.agents.get("content_generator")
        if not generator:
            yield {"event": "failed", "nodeId": expansion_node_id, "content": "错误: 未找到 'content_generator' 代理。"}
            return

        async def expand_perspective(p: Dict, i: int):
            p_node_id = f"{expansion_node_id}_{i}"
            p_title = p.get('title', f'角度 {i+1}')
            yield {"event": "progress", "nodeId": p_node_id, "parentId": expansion_node_id, "title": p_title}
            
            prompt = f"原始问题: {problem}\n\n分析角度: {p_title}\n角度描述: {p.get('description')}"
            
            expanded_content = await generator.invoke_non_stream(prompt)
            
            yield {"event": "completed", "nodeId": p_node_id, "content": expanded_content}
            yield expanded_content

        # 使用我们之前定义的并行生成器运行器
        expansion_generators = [expand_perspective(p, i) for i, p in enumerate(perspectives)]
        expanded_contents = []
        async for item in self._run_generators_in_parallel(expansion_generators):
            if "event" in item:
                yield item
            else:
                expanded_contents.append(item)

        yield {
            "event": "completed", "nodeId": expansion_node_id, "title": "扩展完成",
            "content": f"所有 {len(expanded_contents)} 个角度已扩展完毕。"
        }

        # --- 步骤 3: 总编辑 -> 综合并润色 ---
        editor_node_id = "general_editor"
        yield {
            "event": "progress", "nodeId": editor_node_id, "parentId": parent_node_id, "title": "综合",
            "content": "总编辑正在将所有内容片段整合成一篇完整的文章..."
        }
        
        editor = self.agents.get("editor_in_chief")
        if not editor:
            yield {"event": "failed", "nodeId": editor_node_id, "content": "错误: 未找到 'editor_in_chief' 代理。"}
            return
            
        fragments_for_editor = [f"--- 内容片段 {i+1} ---\n{content}" for i, content in enumerate(expanded_contents)]
        editor_prompt = f"原始问题: {problem}\n\n" + "\n\n".join(fragments_for_editor)
        
        final_response = ""
        async for chunk in editor.invoke(editor_prompt):
            yield {"event": "chunk", "nodeId": editor_node_id, "content": chunk}
            final_response += chunk
            
        yield {
            "event": "completed", "nodeId": editor_node_id, "title": "综合完成",
            "content": final_response
        }


async def main_test():
    """异步测试函数"""
    try:
        orchestrator = Orchestrator(verbose=True)
        problem = "我们如何利用AI技术来改善在线教育的个性化学习体验？"
        
        print("\n" + "="*25 + " 测试顺序模式 (流式) " + "="*25)
        
        async for event in orchestrator.run(problem, num_solutions=1, mode="sequential"):
            print(f"\n--- EVENT ---")
            if event.get("event") == "chunk":
                print(event.get("content"), end="", flush=True)
            else:
                print(f"\n{json.dumps(event, indent=2, ensure_ascii=False)}")
            print("\n-------------")

    except Exception as e:
        import traceback
        print(f"运行Orchestrator时发生错误: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main_test())
