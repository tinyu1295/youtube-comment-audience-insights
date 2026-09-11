from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = [
        line.strip() for line in f
        if line.strip() and not line.startswith(('-e', '#'))
    ]

setup(
    name="SMA-Sentiment-Intelligence-Plugin",
    version="1.0.0",
    author="TIN YU",
    author_email="tinyu1295@gmail.com",
    description="End-to-end sentiment analysis pipeline and Chrome extension for YouTube comments.",
    packages=find_packages(include=['src', 'src.*']),
    python_requires=">=3.10",
    install_requires=install_requires,
)
