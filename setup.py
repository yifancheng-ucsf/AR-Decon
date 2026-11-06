from setuptools import setup, find_packages

# The deconvolution engine, ER-Decon II (core2_decon), is not bundled: it is
# built from https://github.com/Biochemistry-Biophysics-Cheng-Lab/ER-DECON-II
# and must be on PATH -- see README.md, "Installation".
setup(
    name='ardecon',
    version='0.9.0',
    description="A deconvolution pipeline for anisotropic resolution",
    author="Junrui Li",
    author_email="cheng_lab@ucsf.edu",
    packages=find_packages(include=["ardecon", "ardecon.scripts.*"]),
    include_package_data=True,
    package_data={
        'ardecon': [
            'scripts/*',
        ],
    },
    scripts=[
        'ardecon/scripts/applymask.py',
        'ardecon/scripts/dfsc-0.0.1.py',
        'ardecon/scripts/dfsc-avg-plot.py',
        'ardecon/scripts/gen_sphere_mask.py',
        'ardecon/scripts/gen_tf_withPSF.py',
        'ardecon/scripts/ardecon',
        'ardecon/scripts/core2_decon-000.com',
        'ardecon/scripts/param_search',
        'ardecon/scripts/dfsc-combine.py',
    ],
    install_requires=[],
    zip_safe=False,
)
