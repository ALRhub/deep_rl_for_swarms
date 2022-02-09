from setuptools import setup, find_packages

setup(
    name='deep_rl_for_swarms',
    version='',
    packages=[package for package in find_packages()
              if package.startswith('deep_rl_for_swarms')],
    url='',
    license='',
    author='Maximilian Huettenrauch',
    author_email='',
    description='',
    install_requires=['absl-py == 0.2.2',
                      'astor == 0.7.1',
                      'bleach == 3.1.4',
                      'certifi == 2018.4.16',
                      'chardet == 3.0.4',
                      'cloudpickle == 0.5.3',
                      'cycler == 0.10.0',
                      'decorator == 4.3.0',
                      'dill == 0.2.8.2',
                      'fast-histogram == 0.4',
                      'future == 0.16.0',
                      'gast == 0.2.0',
                      'grpcio == 1.13.0',
                      'gym == 0.10.5',
                      'html5lib == 0.9999999',
                      'idna == 2.7',
                      'kiwisolver == 1.0.1',
                      'Markdown == 2.6.11',
                      'matplotlib == 2.2.2',
                      'mpi4py == 3.0.0',
                      'networkx == 2.1',
                      'numpy == 1.14.5',
                      # 'pkg-resources == 0.0.0',
                      'protobuf == 3.6.0',
                      'pyglet == 1.3.2',
                      'pyparsing == 2.2.0',
                      'python-dateutil == 2.7.3',
                      'pytz == 2018.5',
                      'requests == 2.20.0',
                      'scipy == 1.1.0',
                      # 'Shapely == 1.6.4.post1',
                      'six == 1.11.0',
                      'tensorboard == 1.15.0',
                      'tensorflow == 2.5.3',
                      'termcolor == 1.1.0',
                      'urllib3 == 1.24.2',
                      'Werkzeug == 0.15.3'
                      ]
)
