from setuptools import setup, find_packages

setup(
    name="hover_net",
    version="1.0.0",
    description="HoverNet: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images",
    author="vqdang",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "docopt>=0.6.2",
        "future>=0.18.2",
        "imgaug>=0.4.0",
        "matplotlib>=3.3.0",
        "numpy>=1.19.1",
        "opencv-python>=4.3.0.36",
        "pandas>=1.1.0",
        "pillow>=7.2.0",
        "psutil>=5.7.3",
        "scikit-image>=0.17.2",
        "scikit-learn>=0.23.1",
        "scipy>=1.5.2",
        "tensorboard>=2.3.0",
        "tensorboardx>=2.1",
        "termcolor>=1.1.0",
        "tqdm>=4.48.0",
    ],
    entry_points={
        'console_scripts': [
            # 如果有需要暴露的命令行工具，可以在这里添加
            # 'hover-infer=hover_net.run_infer:main', 
        ],
    },
)
