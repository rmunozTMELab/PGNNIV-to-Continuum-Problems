import subprocess

# subprocess.run("python PGNNIV_heterogeneous_problem.py 1000 0", shell=True)
# subprocess.run("python PGNNIV_heterogeneous_problem.py 1000 1", shell=True)

# subprocess.run("python PGNNIV_tensorial.py 1000 0", shell=True)
# subprocess.run("python PGNNIV_tensorial.py 1000 1", shell=True)

subprocess.run("python PGNNIV_nonlinear_problem_P3.py 10000 0", shell=True)
subprocess.run("python PGNNIV_nonlinear_problem_P3.py 100000 1", shell=True)

subprocess.run("python PGNNIV_nonlinear_problem_P4.py 10000 0", shell=True)
subprocess.run("python PGNNIV_nonlinear_problem_P4.py 100000 1", shell=True)