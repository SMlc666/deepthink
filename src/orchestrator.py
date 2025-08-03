import json
import asyncio
import re
from typing import Dict, Any, List
from .config_loader import load_config, get_agent_settings
from .agent import Agent
from .prompts import get_prompt
from .code_executor import execute_code
from . import tools

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

            # 为 critic agent 加载两个不同的 prompt
            if name == 'critic':
                system_prompt = get_prompt('critic')
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
            else:
                system_prompt = get_prompt(name)

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

    async def run(self, problem: str, num_solutions: int = 3, mode: str = "sequential"):
        """
        执行完整的工作流程，现在包含问题路由。
        """
        print(f"\n{'='*20} 开始处理新问题 ({mode} 模式) {'='*20}")
        print(f"问题: {problem}")

        # 阶段 0: 问题路由
        router = self.agents.get("router")
        if not router:
            print("错误: 未找到 'router' 代理。将默认使用工程工作流。")
            problem_type = "ENGINEER"
        else:
            self._log("\n--- 阶段 0: 问题路由 ---")
            # .strip().upper() 对于稳健匹配很重要
            problem_type = (await router.invoke(problem)).strip().upper()
            self._log(f"  - 路由结果: 问题类型被判断为 '{problem_type}'")

        # 根据问题类型选择工作流
        if problem_type == "MATHEMATICIAN":
            final_results = await self._run_mathematical_workflow(problem)
        else:  # 默认为 'ENGINEER' 工作流
            final_results = await self._run_engineering_workflow(problem, num_solutions, mode)

        if not final_results:
            self._log("工作流未能产出最终结果，流程终止。")
            return

        # 最终评审
        final_review = await self._final_review(problem, final_results)
        print("\n\n--- 最终评审结果 ---")
        print(final_review)

        # 汇总Token使用情况
        self._summarize_usage()
        print(f"\n{'='*20} 流程处理结束 {'='*20}")

    async def _run_engineering_workflow(self, problem: str, num_solutions: int, mode: str) -> List[Dict[str, Any]]:
        """
        执行软件工程工作流（生成代码、应用等）。
        """
        if self.verbose:
            print("\n--- 工程工作流执行 ---")
            print(f"将生成 {num_solutions} 个解决方案思路。")

        # 1. 生成思路
        ideas = await self._generate_ideas(problem, num_solutions)
        if not ideas:
            self._log("未能生成任何思路，流程终止。")
            return []

        # 2. 解析思路
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
        except json.JSONDecodeError:
            print("错误: 'solution_generator'未能返回有效的JSON格式。流程终止。")
            if self.verbose:
                print("原始输出:", ideas)
            return []

        if not solutions:
            print("未能从生成器响应中解析出任何解决方案。")
            return []

        # 3. 根据模式选择工作流
        if mode == "batch":
            return await self._run_batch_workflow(solutions)
        else:
            return await self._run_sequential_workflow(solutions)

    async def _run_sequential_workflow(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.verbose:
            print("\n--- 顺序工作流执行 ---")
        final_results = []
        for i, solution in enumerate(solutions):
            print(f"\n--- 正在处理思路 {i+1}: {solution.get('title', '无标题')} ---")
            
            # 3a. 初始批判 (Initial Critique)
            initial_criticism = await self._critique_idea(solution)
            
            # 3b. 执行与迭代批判循环 (Execution and Iterative Critique Loop)
            execution_log = await self._execute_and_critique_loop(solution, initial_criticism)
            
            final_results.append({
                "solution": solution,
                "criticism": initial_criticism, # 保留初始批判
                "execution_log": execution_log
            })
        return final_results

    async def _run_batch_workflow(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.verbose:
            print("\n--- 批处理工作流执行 ---")
        
        # 阶段 2: 并行执行所有思路
        if self.verbose:
            print("\n--- 阶段 2: 并行执行所有思路 ---")
        execution_tasks = [self._execute_idea(s, "") for s in solutions]
        execution_logs = await asyncio.gather(*execution_tasks)
        if self.verbose:
            print("所有思路执行完成。")

        # 阶段 3: 并行批判所有执行结果
        if self.verbose:
            print("\n--- 阶段 3: 并行批判所有执行结果 ---")
        critique_tasks = [self._critique_idea(s, log) for s, log in zip(solutions, execution_logs)]
        criticisms = await asyncio.gather(*critique_tasks)
        if self.verbose:
            print("所有批判完成。")

        # 整合结果
        final_results = [
            {"solution": sol, "criticism": crit, "execution_log": log}
            for sol, crit, log in zip(solutions, criticisms, execution_logs)
        ]
        return final_results

    async def _generate_ideas(self, problem: str, num_solutions: int) -> str:
        if self.verbose:
            print("\n--- 阶段 1: 生成解决方案思路 ---")
        generator = self.agents.get("solution_generator")
        if not generator:
            print("错误: 未找到 'solution_generator' 代理。")
            return ""

        prompt_template = get_prompt("solution_generator")
        formatted_prompt = prompt_template.format(num_solutions=num_solutions)
        
        response = await generator.invoke(problem, system_prompt_override=formatted_prompt)
        if self.verbose:
            print(f"'{generator.name}' 已生成思路。")
        return response

    async def _critique_idea(self, solution: Dict[str, Any], execution_log: str = "") -> str:
        if self.verbose:
            print("  - 正在进行批判性分析...")
        
        if execution_log:
            # 模式: 批判执行日志 (QA)
            critic = self.agents.get("execution_critic")
            if not critic:
                return "错误: 未找到 'execution_critic' 代理。"
            prompt = f"任务: {solution.get('title')}\n\n执行日志:\n{execution_log}"
            response = await critic.invoke(prompt)
            if self.verbose:
                print("  - 执行结果批判完成。")
        else:
            # 模式: 批判初始思路
            critic = self.agents.get("critic")
            if not critic:
                return "错误: 未找到 'critic' 代理。"
            prompt = f"请批判以下解决方案思路：\n\n标题: {solution.get('title')}\n\n描述: {solution.get('description')}\n\n步骤: {solution.get('steps')}"
            response = await critic.invoke(prompt)
            if self.verbose:
                print("  - 初始思路批判完成。")
        
        return response

    async def _execute_and_critique_loop(self, solution: Dict[str, Any], initial_criticism: str) -> str:
        if self.verbose:
            print("  - 进入执行与迭代批判循环...")
        
        executor = self.agents.get("executor")
        if not executor:
            return "错误: 未找到 'executor' 代理。"

        available_libs_str = ", ".join(self.available_libraries)
        # 根据搜索工具的可用性调整提示
        search_instruction = "如果需要外部知识，请使用搜索工具。否则，请编写代码。"
        if not self.search_tools_available:
            search_instruction = "搜索工具当前不可用。请仅依赖你的内部知识和代码执行能力来完成任务。"

        # 初始Prompt包含原始思路和初始批判
        current_prompt = (
            f"任务：基于以下解决方案思路和批判意见，来完成任务。\n\n"
            f"**你可以使用的Python库**: {available_libs_str}\n\n"
            f"--- 原始思路 ---\n标题: {solution.get('title')}\n描述: {solution.get('description')}\n\n"
            f"--- 初始批判意见 ---\n{initial_criticism}\n\n"
            f"请开始你的第一步。分析问题，制定计划。{search_instruction}"
        )

        conversation_history = [f"Orchestrator: {current_prompt}"]
        max_iterations = 8 # 增加迭代次数以支持工具使用

        for i in range(max_iterations):
            self._log(f"  - 执行/批判循环: 第 {i+1}/{max_iterations} 轮")
            
            # 1. 执行者工作
            executor_response = await executor.invoke(current_prompt)
            conversation_history.append(f"Executor:\n{executor_response}")
            self._log(f"  - Executor响应: {executor_response[:200]}...")

            # 2. 检查是代码执行还是工具调用
            tool_match = re.search(r'\[TOOL_REQUEST\]\s*(\{[\s\S]*\})\s*', executor_response, re.DOTALL)
            code_match = re.search(r'```python\n([\s\S]*?)\n```', executor_response, re.DOTALL)

            feedback_for_next_round = ""

            if tool_match:
                self._log("  - 检测到工具调用请求...")
                # 硬性检查：如果工具不可用，则直接返回错误
                if not self.search_tools_available:
                    self._log("  - 警告: AI在搜索工具不可用时仍尝试使用。")
                    feedback_for_next_round = "[TOOL_RESULT]\n错误: 搜索工具当前不可用。你的请求已被拒绝。请尝试使用代码或其他方法解决问题。\n"
                else:
                    try:
                        # 增加健壮性：从Markdown代码块中提取JSON
                        json_str = ""
                        match = re.search(r'\{[\s\S]*\}', tool_match.group(1))
                        if match:
                            json_str = match.group(0)
                        
                        tool_request = json.loads(json_str)
                        if tool_request.get("tool") == "web_search":
                            search_query = tool_request.get("query", "")
                            # 将原始问题和当前解决方案的上下文传递给搜索工作流
                            search_context = f"原始问题: {solution.get('title')}\n查询需求: {search_query}"
                            search_summary = await self._run_search_workflow(search_context)
                            feedback_for_next_round = f"[TOOL_RESULT]\n{search_summary}\n"
                        else:
                            feedback_for_next_round = "[TOOL_RESULT]\n错误: 未知的工具请求。\n"
                    except json.JSONDecodeError:
                        feedback_for_next_round = f"[TOOL_RESULT]\n错误: 工具请求的JSON格式无效。\n"
                
                conversation_history.append(f"Orchestrator:\n{feedback_for_next_round}")
                self._log(f"  - 工具执行结果: {feedback_for_next_round[:200]}...")

            elif code_match:
                self._log("  - 检测到代码块，正在执行...")
                code_to_execute = code_match.group(1)
                execution_result = execute_code(code_to_execute)
                self._log(f"  - 代码执行完成，返回码: {execution_result['return_code']}")
                execution_result_str = f"[EXECUTION_RESULT]\nSTDOUT:\n{execution_result['stdout']}\n\nSTDERR:\n{execution_result['stderr']}\n"
                conversation_history.append(f"CodeExecutor:\n{execution_result_str}")
                self._log(execution_result_str)

                # 对代码执行结果进行批判
                self._log("  - 正在进行执行结果批判...")
                execution_criticism = await self._critique_idea(solution, execution_result_str)
                conversation_history.append(f"ExecutionCritic:\n{execution_criticism}")
                
                if "[ACCEPTABLE]" in execution_criticism:
                    self._log("  - ExecutionCritic发出 [ACCEPTABLE] 信号，循环结束。")
                    break
                
                feedback_for_next_round = f"{execution_result_str}\n这是对你执行结果的批判意见:\n{execution_criticism}\n\n"

            else:
                self._log("  - 未在响应中找到可执行代码或工具调用。")
                feedback_for_next_round = "你的回复中既没有代码块也没有工具调用。请根据任务，决定是使用工具还是编写代码。\n"

            # 3. 检查执行者是否认为任务完成
            if "[COMPLETE]" in executor_response:
                self._log("  - Executor发出 [COMPLETE] 信号。")
                break
            
            # 4. 检查是否达到最大迭代次数
            if i == max_iterations - 1:
                self._log("  - 达到最大迭代次数，循环强制结束。")
                conversation_history.append("Orchestrator: 达到最大迭代次数，流程终止。")
                break

            # 5. 构建下一轮的Prompt
            current_prompt = (
                f"这是你之前的尝试:\n{executor_response}\n\n"
                f"这是来自系统或工具的反馈:\n{feedback_for_next_round}\n\n"
                "请分析反馈，修正你的计划，然后继续下一步。"
            )

        final_log = "\n\n---\n".join(conversation_history)
        if self.verbose:
            print("  - 执行与迭代批判循环结束。")
        return final_log

    async def _run_search_workflow(self, query_context: str) -> str:
        """
        执行完整的搜索->抓取->总结工作流。
        """
        self._log("\n--- 搜索工作流启动 ---")
        
        searcher = self.agents.get("searcher")
        summarizer = self.agents.get("summarizer")
        if not searcher or not summarizer:
            return "错误: 未找到 'searcher' 或 'summarizer' 代理。"

        # 1. 使用 Searcher 代理生成搜索查询
        self._log(f"  - Searcher正在处理查询上下文: {query_context}")
        searcher_response = await searcher.invoke(query_context)
        self._log(f"  - Searcher响应: {searcher_response}")
        
        try:
            # 增加健壮性：从Markdown代码块中提取JSON
            json_str = ""
            match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', searcher_response)
            if match:
                json_str = match.group(1)
            else:
                # 如果没有找到Markdown块，就假设整个响应是JSON
                json_str = searcher_response

            search_request = json.loads(json_str)
            search_query = search_request.get("query")
            if not search_query:
                return "错误: Searcher未能生成有效的搜索查询。"
        except json.JSONDecodeError:
            return f"错误: Searcher返回了无效的JSON: {searcher_response}"

        # 2. 执行Bing搜索
        self._log(f"  - 正在执行Bing搜索: '{search_query}'")
        search_results = tools.bing_search(search_query, num_results=10) # 获取前10个结果
        if not search_results or "error" in search_results[0]:
            return f"错误: 必应搜索失败: {search_results}"
        
        self._log(f"  - 找到 {len(search_results)} 个搜索结果。")

        # 3. 并行抓取网页内容
        self._log("  - 正在并行抓取网页内容...")
        fetch_tasks = [tools.fetch_webpage_content(res["link"]) for res in search_results if res.get("link")]
        web_contents = await asyncio.gather(*fetch_tasks)
        
        combined_content = "\n\n--- 网页分割线 ---\n\n".join(web_contents)
        self._log(f"  - 内容抓取完成，总长度: {len(combined_content)} 字符。")

        # 4. 使用 Summarizer 代理总结内容
        self._log("  - Summarizer正在总结内容...")
        summarizer_prompt = (
            f"**原始查询上下文:**\n{query_context}\n\n"
            f"**以下是为你抓取的网页内容，请根据原始查询上下文进行总结:**\n\n{combined_content}"
        )
        summary = await summarizer.invoke(summarizer_prompt)
        self._log("  - 总结完成。")
        
        return summary

    async def _run_mathematical_workflow(self, problem: str) -> List[Dict[str, Any]]:
        if self.verbose:
            print("\n--- 数学工作流执行 ---")

        # 对于数学工作流，我们将整个问题视为一个待解决的“方案”
        solution = {
            "title": f"数学问题: {problem[:50]}...",
            "description": "使用符号计算和逻辑推导来解决该问题。",
            "steps": ["形式化问题", "使用sympy进行推导", "得出结论"]
        }

        print(f"\n--- 正在处理数学问题: {solution.get('title', '无标题')} ---")

        # 数学家工作流直接进入执行与推导循环
        execution_log = await self._execute_and_critique_loop_for_math(problem, solution)

        final_results = [{
            "solution": solution,
            "criticism": "", # 数学工作流的批判意见已包含在执行日志中
            "execution_log": execution_log
        }]
        return final_results

    async def _execute_and_critique_loop_for_math(self, problem: str, solution: Dict[str, Any]) -> str:
        if self.verbose:
            print("  - 进入数学执行与批判循环...")

        mathematician = self.agents.get("mathematician")
        math_critic = self.agents.get("math_critic") # 获取新的批判家代理
        if not mathematician or not math_critic:
            return "错误: 未找到 'mathematician' 或 'math_critic' 代理。"

        available_libs_str = ", ".join(self.available_libraries)
        search_instruction = "如果需要外部知识，请使用搜索工具。否则，请使用 `sympy`, `numpy`, `scipy` 编写代码来辅助你的证明。"
        if not self.search_tools_available:
            search_instruction = "搜索工具当前不可用。请仅依赖你的内部知识和代码执行能力来完成任务。"
        
        current_prompt = (
            f"任务：解决以下数学问题。\n\n"
            f"**你可以使用的Python库**: {available_libs_str}\n"
            f"**问题**: {problem}\n\n"
            f"请开始你的第一步。分析问题，制定计划。{search_instruction}"
        )

        conversation_history = [f"Orchestrator: {current_prompt}"]
        max_iterations = 15

        for i in range(max_iterations):
            self._log(f"  - 数学循环: 第 {i+1}/{max_iterations} 轮")

            # 1. 数学家工作
            mathematician_response = await mathematician.invoke(current_prompt)
            conversation_history.append(f"Mathematician:\n{mathematician_response}")
            self._log(f"  - Mathematician响应: {mathematician_response[:200]}...")

            tool_match = re.search(r'\[TOOL_REQUEST\]\s*(\{[\s\S]*\})\s*', mathematician_response, re.DOTALL)
            code_match = re.search(r'```python\n([\s\S]*?)\n```', mathematician_response, re.DOTALL)

            feedback_for_next_round = ""

            if tool_match:
                self._log("  - 检测到工具调用请求...")
                if not self.search_tools_available:
                    feedback_for_next_round = "[TOOL_RESULT]\n错误: 搜索工具当前不可用。你的请求已被拒绝。请尝试使用代码或其他方法解决问题。\n"
                else:
                    try:
                        json_str = re.search(r'\{[\s\S]*\}', tool_match.group(1)).group(0)
                        tool_request = json.loads(json_str)
                        if tool_request.get("tool") == "web_search":
                            search_query = tool_request.get("query", "")
                            search_context = f"原始问题: {solution.get('title')}\n查询需求: {search_query}"
                            search_summary = await self._run_search_workflow(search_context)
                            feedback_for_next_round = f"[TOOL_RESULT]\n{search_summary}\n"
                        else:
                            feedback_for_next_round = "[TOOL_RESULT]\n错误: 未知的工具请求。\n"
                    except (json.JSONDecodeError, AttributeError):
                        feedback_for_next_round = f"[TOOL_RESULT]\n错误: 工具请求的JSON格式无效。\n"
                
                conversation_history.append(f"Orchestrator:\n{feedback_for_next_round}")
                self._log(f"  - 工具执行结果: {feedback_for_next_round[:200]}...")

            elif code_match:
                self._log("  - 检测到代码块，正在执行...")
                code_to_execute = code_match.group(1)
                execution_result = execute_code(code_to_execute)
                self._log(f"  - 代码执行完成，返回码: {execution_result['return_code']}")
                execution_result_str = f"[EXECUTION_RESULT]\nSTDOUT:\n{execution_result['stdout']}\n\nSTDERR:\n{execution_result['stderr']}\n"
                conversation_history.append(f"CodeExecutor:\n{execution_result_str}")
                self._log(execution_result_str)

                # 2. 数学批判家审查
                self._log("  - 正在进行数学批判...")
                # 批判家的输入是数学家的完整响应 + 执行结果
                critic_prompt = f"这是数学家的工作和代码执行结果，请审查：\n\n{mathematician_response}\n\n{execution_result_str}"
                criticism = await math_critic.invoke(critic_prompt)
                conversation_history.append(f"MathCritic:\n{criticism}")
                self._log(f"  - MathCritic响应: {criticism}")

                if "[ACCEPTABLE]" in criticism:
                    self._log("  - MathCritic发出 [ACCEPTABLE] 信号，循环结束。")
                    break
                
                feedback_for_next_round = f"{execution_result_str}\n这是对你工作的批判意见:\n{criticism}\n\n"

            else:
                self._log("  - 未在响应中找到可执行代码或工具调用。")
                feedback_for_next_round = "你的回复中既没有代码块也没有工具调用。请根据任务，决定是使用工具还是编写代码。\n"

            if "[COMPLETE]" in mathematician_response:
                self._log("  - Mathematician发出 [COMPLETE] 信号。")
                break
            
            if i == max_iterations - 1:
                self._log("  - 达到最大迭代次数，循环强制结束。")
                conversation_history.append("Orchestrator: 达到最大迭代次数，流程终止。")
                break

            current_prompt = (
                f"这是你之前的尝试:\n{mathematician_response}\n\n"
                f"这是来自系统或工具的反馈:\n{feedback_for_next_round}\n\n"
                "请分析反馈，修正你的计划，然后继续下一步。"
            )

        final_log = "\n\n---\n".join(conversation_history)
        if self.verbose:
            print("  - 数学执行与批判循环结束。")
        return final_log

    async def _final_review(self, problem: str, results: List[Dict[str, Any]]) -> str:
        if self.verbose:
            print("\n--- 阶段 4: 最终评审 ---")
        reviewer = self.agents.get("final_reviewer")
        if not reviewer:
            return "错误: 未找到 'final_reviewer' 代理。"

        context = f"原始问题: {problem}\n\n"
        for i, result in enumerate(results):
            context += f"--- 方案 {i+1}: {result['solution'].get('title')} ---\n"
            context += f"思路描述: {result['solution'].get('description')}\n"
            context += f"批判意见: {result['criticism']}\n"
            context += f"执行日志: \n---\n{result['execution_log']}\n---\n\n"
            
        prompt = f"请基于以下所有信息，进行最终的、全面的评审，为每个方案打分，并推荐最佳方案。\n\n{context}"
        response = await reviewer.invoke(prompt)
        if self.verbose:
            print("  - 最终评审完成。")
        return response

    def _summarize_usage(self):
        """
        汇总并打印所有代理的token使用情况。
        """
        print("\n--- Token 使用情况汇总 ---")
        for agent in self.agents.values():
            usage = agent.get_usage_stats()
            self.total_usage["prompt_tokens"] += usage["prompt_tokens"]
            self.total_usage["completion_tokens"] += usage["completion_tokens"]
            self.total_usage["total_tokens"] += usage["total_tokens"]
            print(f"代理 '{agent.name}': {usage}")
        
        print(f"\n总计: {self.total_usage}")


async def main_test():
    """异步测试函数"""
    try:
        orchestrator = Orchestrator()
        problem = "我们如何利用AI技术来改善在线教育的个性化学习体验？"
        
        print("\n" + "="*25 + " 测试顺序模式 " + "="*25)
        await orchestrator.run(problem, mode="sequential")
        
        # 重置 token 计数器以便清晰地看到每种模式的消耗
        orchestrator.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for agent in orchestrator.agents.values():
            agent.total_prompt_tokens = 0
            agent.total_completion_tokens = 0

        print("\n" + "="*25 + " 测试批处理模式 " + "="*25)
        await orchestrator.run(problem, mode="batch")

    except Exception as e:
        print(f"运行Orchestrator时发生错误: {e}")

if __name__ == '__main__':
    # 这是一个简单的测试，用于验证Orchestrator的基本功能
    # 在运行此测试之前，请确保你已经：
    # 1. 创建了 .env 文件并填入了你的 OPENAI_API_KEY。
    # 2. 运行 `pip install -r requirements.txt` 安装了所有依赖。
    asyncio.run(main_test())