"""
此模块定义了应用程序特定的自定义异常。
"""

class EmptyApiResponseError(Exception):
    """当API调用成功但返回空响应时引发的异常。"""
    pass
