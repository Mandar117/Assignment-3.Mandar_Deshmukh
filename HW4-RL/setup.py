from setuptools import setup, find_packages

setup(
    name="hw4_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "gymnasium",
        "scipy",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="HW4-RL: Reinforcement Learning Implementation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/HW4-RL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 