1. Clone all the github repositories in the project into your src folder
2. Navigate to dev_ws/src/random_nav/random_nav and clone the github reposity
3. Navigate to dev_ws/src/random_nav and open the setup.py file
4. Find the entry point line and change it to:

entry_points={
    'console_scripts': [
        'random_nav = random_nav.final_project_theo_allie_jason_ella.random_nav:main',
    ],
},

5. Rebuild the random nav package using:

	colcon build --symlink-install --packages-select random_nav
6. All done!
