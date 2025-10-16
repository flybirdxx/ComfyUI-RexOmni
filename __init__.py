# 导入节点类
from .rex_omni_nodes import (
    RexOmniLoader, 
    RexOmniDetector
)

# 定义节点映射
NODE_CLASS_MAPPINGS = {
    "RexOmniLoader": RexOmniLoader,
    "RexOmniDetector": RexOmniDetector,
}

# 定义节点在UI中的显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "RexOmniLoader": "Rex-Omni Loader",
    "RexOmniDetector": "Rex-Omni Detector",
}

# 打印日志，确认插件已加载
print("------------------------------------------")
print("Rex-Omni Custom Nodes Loaded.")
print("------------------------------------------")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
