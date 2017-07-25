from setuptools import setup, find_packages
from version import write_version_file

# use pip --editable to create symlink :):):)   
# mkdocs
# look at mau mau
#def read(fname):
#        return open(os.path.join(os.path.dirname(__file__), fname)).read()

requirements = ["pytest", "pandas", "numpy", "matplotlib"]
setup(
            name = "Dussianpc",
            version =i write_version_file(),
            author = "Pierre Springer",
            author_email = "pierre.springer@tum.de",
            description = ("""Simple simulation of gaussian processes
                           with some virtualization""" ),
            license = "MIT",
            #keywords = "example documentation tutorial",
            #url = "http://packages.python.org/an_example_pypi_project",
            packages = find_packages(), #  Use package names,
            install_requires = requirements, # Use package name without version
            # long_descrip  tion=read('README'),
            classifiers=[
                        "Development Status :: 3 - Alpha",
                        "Topic :: Utilities",
                        "License :: OSI Approved :: BSD License",
                    ],
    )

