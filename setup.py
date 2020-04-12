
import setuptools

setuptools.setup(
    name="deeparc2whitebalance",
    version="0.0.1",
    author="Vision and Learning lab",
    author_email="allist@vistec.ac.th",
    description="do white balance for image that capture with deeparc",
    url="https://github.com/pureexe/deeparc2whitebalance",
    packages=[''],
    py_modules=['deeparc2whitebalance'],
    install_requires=[
          'numpy',
          'opencv-python',
          'matplotlib'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
     'console_scripts': ['deeparc2whitebalance=deeparc2whitebalance:entry_point'],
    },
    python_requires='>=3.6'
)