#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import re
from glob import glob
from os import getenv
from os.path import abspath
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME
import torch


def get_extensions(extensions_dir, extension_name):
    """
    Code for getting extensions taken from
    https://github.com/LUCKMOONLIGHT/SLRDet/blob/master/setup_rotated.py
    """
    this_dir = dirname(abspath(__file__))
    extensions_dir = join(this_dir, extensions_dir)

    main_file = glob(join(extensions_dir, "*.cpp"))
    source_cpu = glob(join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob(join(extensions_dir, "cuda", "*.cu"))
    print(extensions_dir)
    cnine_cuda_ops_file = join(extensions_dir, "../../backend/cnine/v1/cuda", "*.cu")
    cnine_cuda_ops = glob(join(extensions_dir, "../../backend/cnine/v1/cuda", "*.cu"))
    print("cnine_cuda_ops_file")
    print(cnine_cuda_ops_file)
    print("cnine_cuda_ops")
    print(cnine_cuda_ops)

    sources = main_file + source_cpu + cnine_cuda_ops
    print("!!!!!!!!!!!!!!!!!!")
    print(sources)
    print("!!!!!!!!!!!!!!!!!!")
    extension = CppExtension

    extra_compile_args = {"cxx": ["-lstdc++", "-lm", "-lpthread", "-Wno-sign-compare", "-Wno-unused-variable", "-Wno-reorder", "-Wno-deprecated-declarations"]}
    # extra_compile_args = {"cxx": ["-lstdc++", "-lm", "-lpthread", "-w"]}
    define_macros = []

    print(CUDA_HOME)
    print(sources)
    print("!!!!!!!!!!!!!!!!!!")
    if (torch.cuda.is_available() and CUDA_HOME is not None) or getenv("FORCE_CUDA", "0") == "1":
        print("CUDA IS AVAILABLE")
        extension = CUDAExtension
        print(source_cuda)
        sources += source_cuda
        print(sources)
        print("@@@@@@@@@@@@@@@@@@")
        # sources.append(extensions_dir + "/../../backend/cnine/v1/include/Cnine_base.cu")
        # sources.append(extensions_dir + "/../../backend/GElib/v2/cuda/SO3partA_CGproduct.cu")
        print(sources)
        print("@@@@@@@@@@@@@@@@@@ end")
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-lcublas",
            "-lcudadevrt",
            # "-DCUDA_HAS_FP16=1",
            # "-D__CUDA_NO_HALF_OPERATORS__",
            # "-D__CUDA_NO_HALF_CONVERSIONS__",
            # "-D__CUDA_NO_HALF2_OPERATORS__",
            "-O3",
            # "-DNDEBUG",
            "--use_fast_math",
            "-D_WITH_CUDA",
            "-D_WITH_CUBLAS",
            "-m64",
            # "-rdc=true"
        ]

    print(sources)
    sources = [join(extensions_dir, s) for s in sources]
    print("post join: ", sources)
    print(extensions_dir)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    include_dirs = [extensions_dir,
                    extensions_dir + '/../../backend/GElib/v2/include',
                    extensions_dir + '/../../backend/GElib/v2/objects/SO3',
                    extensions_dir + '/../../backend/GElib/v2/objects/SO3/cell_ops',
                    extensions_dir + '/../../backend/GElib/v2/objects/SO3/cell_ops',
                    # extensions_dir + '/../../backend/GElib/v2/cuda',
                    extensions_dir + '/../../backend/cnine/v1/include',
                    extensions_dir + '/../../backend/cnine/v1/objects/scalar',
                    extensions_dir + '/../../backend/cnine/v1/objects/tensor',
                    extensions_dir + '/../../backend/cnine/v1/objects/tensor_array',
                    extensions_dir + '/../../backend/cnine/v1/objects/tensor_array/cell_ops',
                    # extensions_dir + '/../../backend/cnine/v1/cuda/'
                    ]
    print("extension:", extension)
    print("sources:", sources)
    print("include_dirs:", include_dirs)
    print("define_macros:", define_macros)
    print("extra_compile_args:", extra_compile_args)
    ext_modules = [
        extension(
            extension_name,
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            runtime_libraries=['cudart'],
            libraries=['cudart', 'dl'],
            extra_compile_args=extra_compile_args,
            # depends=[extensions_dir + "/../../backend/GElib/v2/cuda/SO3partA_CGproduct.cu"]
            depends=[extensions_dir + "/../../backend/cnine/v1/include/Cnine_base.cu"]
        )
    ]

    return ext_modules


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


if __name__ == '__main__':

    extensions_dir = "src/pygelib"
    extension_name = "pygelib_cpp"
    ext_modules = []
    ext_modules += get_extensions(extensions_dir, extension_name)

    setup(
        name='pygelib',
        version='0.0.0',
        license='MIT',
        description='Python interface to GElib',
        long_description='%s\n%s' % (
            re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
            re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
        ),
        author='Erik Henning Thiede',
        author_email='ehthiede@gmail.com',
        url='https://github.com/ehthiede/python-pygelib',
        packages=find_packages('src'),
        package_dir={'': 'src'},
        py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
        include_package_data=True,
        zip_safe=False,
        classifiers=[
            # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: MIT License',
            'Operating System :: Unix',
            'Operating System :: POSIX',
            'Operating System :: Microsoft :: Windows',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            # uncomment if you test on these interpreters:
            # 'Programming Language :: Python :: Implementation :: IronPython',
            # 'Programming Language :: Python :: Implementation :: Jython',
            # 'Programming Language :: Python :: Implementation :: Stackless',
            'Topic :: Utilities',
        ],
        project_urls={
            'Documentation': 'https://python-pygelib.readthedocs.io/',
            'Changelog': 'https://python-pygelib.readthedocs.io/en/latest/changelog.html',
            'Issue Tracker': 'https://github.com/ehthiede/python-pygelib/issues',
        },
        keywords=[
            # eg: 'keyword1', 'keyword2', 'keyword3',
        ],
        python_requires='>=3.7',
        install_requires=[
            # eg: 'aspectlib==1.1.1', 'six>=1.7',
        ],
        extras_require={
            # eg:
            #   'rst': ['docutils>=0.11'],
            #   ':python_version=="2.6"': ['argparse'],
        },
        ext_modules=ext_modules,
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    )
    # blahblah
