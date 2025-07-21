

# 添加内存监控
def print_memory_usage():
    """打印当前内存使用情况"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")