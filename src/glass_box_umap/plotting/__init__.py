from .check import check_package, get_plotting_requirements

for req in get_plotting_requirements():
    check_package(req)
