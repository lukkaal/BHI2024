import subprocess

try:
    # 指定 conda.sh 路径
    conda_sh_path = "/home/pci/anaconda3/etc/profile.d/conda.sh"
    command = f"source {conda_sh_path} && conda activate luka_bhi && python /home/pci/luka_bhi/nineth.py"

    # 使用 subprocess 运行命令
    result = subprocess.run(
        ['bash', '-c', command],  # 在 bash shell 中执行命令
        capture_output=True,      # 捕获标准输出和错误
        text=True,                # 输出为字符串格式
        check=True                # 如果命令失败则抛出异常
    )

    # 获取脚本输出
    output = result.stdout
    print("Script output:", output)
except subprocess.CalledProcessError as e:
    print(f"Error running the script: {e.stderr}")
    raise RuntimeError(f"Script execution failed: {e.stderr}")