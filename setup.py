
import setuptools

setuptools.setup(
    name="deeparc2whitebalance",
    version="0.0.1",
    author="Vision and Learning lab",
    author_email="allist@vistec.ac.th",
    description="convert images from deeparc into colmap undistorted",
    url="https://github.com/pureexe/deeparc2undistort",
    packages=[''],
    py_modules=['deeparc2whitebalance'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux 64 Bit",
    ],
    entry_points={
     'console_scripts': ['deeparc2whitebalance=deeparc2whitebalance:entry_point'],
    },
    python_requires='>=3.6'
)