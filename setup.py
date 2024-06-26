from setuptools import setup, find_packages

# packaging==23.2, torch==2.0.1 따로 설치 필요
setup(
    name='ragcar', 
    version='0.1.0', 
    url='https://github.com/leewaay/ragcar.git', 
    author='Wonseok Lee', 
    author_email='wonsuklee7020@gmail.com', 
    description='RAGCAR: Retrieval-Augmented Generative Companion for Advanced Research', 
    packages=find_packages(), 
    python_requires='>=3.8',
    install_requires=[
        'torch==2.0.1',
        'packaging==23.2',
        'python-dateutil',
        'dataclasses_json',
        'python-dotenv',
        'tqdm',
        'pandas',
        'gdown',
        'tiktoken',
        'kiwipiepy',
        'elasticsearch==7.13.1',
        'aiohttp',
        'sentence-transformers==2.4.0',
        'openai==0.28.1',
        'pytorch-lightning==1.1.0',
        'fsspec>=2021.4.0,<=2023.5.0',
        'transformers @ git+https://github.com/huggingface/transformers.git@bebeeee01275c32fccec3fa36d8b148d3813a7dc',
        'peft @ git+https://github.com/huggingface/peft.git',
        'deepspeed>=0.13.1',
        'xformers==0.0.22',
        'flash-attn==2.3.3',
        'bitsandbytes>=0.43.0',
        'sentencepiece',
        'addict'
    ],
)