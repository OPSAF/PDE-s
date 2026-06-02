@echo off
call "C:\Users\27862\anaconda3\Scripts\activate.bat"
call conda activate C:\Users\27862\anaconda3\envs\P
python ex_pinn.py --epochs 10000%*
pause