#!/bin/bash
# ====================================================
# 脚本名称: unzip.sh
# 功能: 解压当前目录下所有带 Light1~Light4 的 zip 文件
#       分别解压到 Light1-raw, Light2-raw, Light3-raw, Light4-raw 子目录
# 使用方法: 将此脚本放在 48AMA/imgs/S26F30091-09/ 下，
#           执行 ./unzip.sh (或 bash unzip.sh)
# 作者: (可选)
# 日期: 2026-07-21
# ====================================================

# 切换到脚本所在目录（保证相对路径正确）
cd "$(dirname "$0")" || { echo "错误: 无法切换到脚本目录"; exit 1; }

# 检查 unzip 命令是否可用
if ! command -v unzip &> /dev/null; then
    echo "错误: 未找到 unzip 命令，请先安装 unzip。"
    exit 1
fi

# 开启 nullglob，避免无匹配文件时通配符保留字面量
shopt -s nullglob

# 定义要处理的 Light 编号
lights=(1 2 3 4)

# 遍历每个编号
for num in "${lights[@]}"; do
    target_dir="Light${num}-raw"
    echo "处理 Light${num} ..."
    # 创建目标目录（如果不存在）
    mkdir -p "$target_dir"
    # 收集匹配的 zip 文件
    zip_files=( *Light${num}*.zip )
    if [ ${#zip_files[@]} -eq 0 ]; then
        echo "  警告: 没有找到包含 Light${num} 的 zip 文件，跳过。"
        continue
    fi
    # 解压每个 zip 文件到目标目录
    for zipfile in "${zip_files[@]}"; do
        echo "  正在解压: $zipfile -> $target_dir/"
        unzip -o "$zipfile" -d "$target_dir/"
        if [ $? -ne 0 ]; then
            echo "    错误: 解压 $zipfile 失败，请检查文件完整性。"
        fi
    done
    echo "Light${num} 处理完成。"
done

echo "所有 Light 解压任务完成！"