import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox

def remove_comments_and_extra_blank_lines(code: str) -> str:
    # 删除 /* ... */ 块注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # 删除 // 单行注释
    code = re.sub(r'//.*', '', code)
    # 去掉行尾多余空白
    code = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)
    # 删除多余空行
    code = re.sub(r'\n\s*\n+', '\n\n', code)
    return code.strip() + '\n'

def process_file(filepath: str):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            original = f.read()
        cleaned = remove_comments_and_extra_blank_lines(original)
        if cleaned != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            print(f"[OK] Cleaned: {filepath}")
        else:
            print(f"[SKIP] No change: {filepath}")
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")

def process_folder(root_dir: str):
    total = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(('.cpp', '.h')):
                process_file(os.path.join(dirpath, fn))
                total += 1
    return total

def main():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    folder = filedialog.askdirectory(title="请选择要清理的源码文件夹")
    if not folder:
        messagebox.showinfo("取消", "未选择文件夹，操作已取消。")
        return

    print(f"\n🔍 正在处理目录：{folder}\n")
    count = process_folder(folder)
    print("\n✅ 处理完成！共检查文件数量：", count)
    messagebox.showinfo("完成", f"处理完成！共检查 {count} 个 .cpp / .h 文件。")

if __name__ == "__main__":
    main()
