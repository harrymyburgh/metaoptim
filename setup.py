from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='metaoptim',
    version='0.0.9',
    description='A Python package for metaheuristic optimisation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/harrymyburgh/metaoptim',
    project_urls={
        'Bug Tracker': 'https://github.com/harrymyburgh/metaoptim/issues'
    },
    author='Harry Phillip Myburgh',
    author_email='harry.myburgh@gmail.com',
    py_modules=['metaoptim'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
        'numpy >= 1.23.0',
        'numba >= 0.56.0',
        'tqdm >= 4.64.0',
    ],
    include_package_data=True,
)



