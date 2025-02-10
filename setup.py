from setuptools import setup, find_packages

lib_dir = "ardecon/lib"
so_files = [os.path.join(lib_dir, f) for f in os.listdir(lib_dir) if ".so" in f]

setup(
    name='ardecon',
    version='0.8.0',
    description="A deconvolution pipeline for anisotropic resolution",
    author="Junrui Li",
    author_email="cheng_lab@ucsf.edu",
    packages=find_packages(include=["ardecon", "ardecon.scripts.*"]),
    include_package_data=True,
    package_data={
        'ardecon': [
            'bin/*',
            'lib/*',
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
    data_files=[
        ('lib', so_files),
        ('bin', ['ardecon/bin/core2_decon','ardecon/bin/core2_decon_static']),
        ('etc/conda/activate.d', ['etc/conda/activate.d/set_env.sh']),
        ('etc/conda/deactivate.d', ['etc/conda/deactivate.d/unset_env.sh']),
    ],
    install_requires=[],
    zip_safe=False,
)
