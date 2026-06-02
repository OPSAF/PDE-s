@echo off
powershell -Command "
    # 初始化conda（如果需要）
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        & 'C:\Users\27862\anaconda3\shell\condabin\conda-hook.ps1'
    }
    
    # 激活虚拟环境
    conda activate C:\Users\27862\anaconda3\envs\P
    
    # 运行PINN脚本（可根据需要添加参数）
    python ex_pinn.py --10000%*
"