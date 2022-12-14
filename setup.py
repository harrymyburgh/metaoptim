from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='open-metaoptim',
    version='0.0.13',
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
    package_dir={'': 'src'},
    packages=[
        'metaoptim',
        'metaoptim.pso',
    ],
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
    extras_require={
        'dev': [
            'pytest >= 7.2.0',
            'check-manifest >= 0.49',
            'twine >= 4.0.2',
            'pycodestyle >= 2.10.0',
        ]
    },
    include_package_data=True,
)



