from setuptools import setup
import os

setup(
    name='ibopf',
    version='0.1',
    packages=['ibopf', 'ibopf.feature_extraction', 'ibopf.decomposition',
              'ibopf.feature_selection', 'ibopf.neighbors'],
    url='https://github.com/frmunozz/irregular-bag-of-pattern',
    license='MIT',
    author='Francisco Muñoz',
    author_email='fjmunoz95@gmail.com',
    description='Irregular Bag-of-Pattern Feature method',
    data_files=[('', ['settings.json'])],
    scripts=['scripts/' + f for f in os.listdir('scripts')],
)
