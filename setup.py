from setuptools import setup

setup(
    name='ibopf',
    version='0.1',
    packages=['src', 'src.mmmbopf', 'src.neighbors', 'src.Adeprecated', 'src.Adeprecated.sax', 'src.Adeprecated.bopf',
              'src.Adeprecated.core', 'src.Adeprecated.mmtva', 'src.Adeprecated.methodA', 'src.Adeprecated.transformer',
              'src.Adeprecated.representation', 'src.decomposition', 'src.feature_selection', 'src.feature_extraction',
              'src.feature_extraction.text'],
    url='https://github.com/frmunozz/irregular-bag-of-pattern',
    license='',
    author='Francisco Muñoz',
    author_email='fjmunoz95@gmail.com',
    description='Irregular Bag-of-Pattern Feature method'
)
