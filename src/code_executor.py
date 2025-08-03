import subprocess
import sys
from typing import Dict, Any

def execute_code(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    在一个隔离的子进程中安全地执行Python代码字符串。
    使用 Popen 以获得更健壮的I/O捕获。

    Args:
        code (str): 要执行的Python代码。
        timeout (int): 执行的超时时间（秒）。

    Returns:
        一个字典，包含:
        - 'stdout' (str): 执行的标准输出。
        - 'stderr' (str): 执行的标准错误。
        - 'return_code' (int): 进程的返回码 (0 表示成功)。
    """
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # 捕获原始字节流，然后在主进程中解码
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
        
        stdout = stdout_bytes.decode('utf-8', errors='replace')
        stderr = stderr_bytes.decode('utf-8', errors='replace')

        return {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": process.returncode,
        }
    except subprocess.TimeoutExpired:
        process.kill()
        # 再次调用 communicate 以获取残留的输出
        stdout_bytes, stderr_bytes = process.communicate()
        
        stdout = stdout_bytes.decode('utf-8', errors='replace')
        stderr = stderr_bytes.decode('utf-8', errors='replace')
        
        return {
            "stdout": stdout,
            "stderr": f"执行超时：代码运行超过 {timeout} 秒。\n{stderr}",
            "return_code": -1,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"执行代码时发生未知错误: {e}",
            "return_code": -1,
        }

if __name__ == '__main__':
    # 用于测试模块功能的代码
    print("--- 测试 code_executor ---")

    # 1. 测试成功执行
    print("\n1. 测试成功执行...")
    success_code = "import os; print('Hello from subprocess!'); print(f'Current directory: {os.getcwd()}')"
    result = execute_code(success_code)
    print(f"返回码: {result['return_code']}")
    print(f"标准输出:\n{result['stdout']}")
    print(f"标准错误:\n{result['stderr']}")
    assert result['return_code'] == 0
    assert "Hello from subprocess!" in result['stdout']

    # 2. 测试执行出错
    print("\n2. 测试执行出错...")
    error_code = "print('Start'); import non_existent_module; print('End')"
    result = execute_code(error_code)
    print(f"返回码: {result['return_code']}")
    print(f"标准输出:\n{result['stdout']}")
    print(f"标准错误:\n{result['stderr']}")
    assert result['return_code'] != 0
    assert "ModuleNotFoundError" in result['stderr']

    # 3. 测试超时
    print("\n3. 测试超时...")
    timeout_code = "import time; print('Sleeping...'); time.sleep(5); print('Woke up!')"
    result = execute_code(timeout_code, timeout=2)
    print(f"返回码: {result['return_code']}")
    print(f"标准输出:\n{result['stdout']}")
    print(f"标准错误:\n{result['stderr']}")
    assert result['return_code'] == -1
    assert "执行超时" in result['stderr']
    
    print("\n--- 所有测试完成 ---")