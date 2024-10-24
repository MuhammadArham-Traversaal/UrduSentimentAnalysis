pip install unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps packaging ninja einops "flash-attn>=2.6.3"
pip install triton
pip install xformers --no-cache-dir
pip install pandas numpy matplotlib seaborn scikit-learn