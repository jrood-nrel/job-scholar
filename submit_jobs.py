#!/usr/bin/env python

"""Job Scholar highest level submission script.
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import os
import yaml
import argparse
import datetime
import stat
import subprocess
import shutil


# ========================================================================
#
# Classes
#
# ========================================================================


class Job:

    def __init__(self,
                 name,
                 queue,
                 mapping,
                 compiler,
                 executable,
                 spec,
                 input_file,
                 mesh,
                 files_to_copy,
                 nodes,
                 minutes,
                 pre_args,
                 post_args,
                 awind_ranks,
                 nwind_ranks,
                 walltime,
                 ranks_per_node,
                 gpus_per_node,
                 hyperthreads,
                 cpu_bind,
                 total_ranks,
                 total_gpus,
                 path,
                 script,
                 profile):

        self.name = name
        self.queue = queue
        self.mapping = mapping
        self.compiler = compiler
        self.executable = executable
        self.spec = spec
        self.input_file = input_file
        self.mesh = mesh
        self.files_to_copy = files_to_copy
        self.nodes = nodes
        self.minutes = minutes
        self.pre_args = pre_args
        self.post_args = post_args
        self.awind_ranks = awind_ranks
        self.nwind_ranks = nwind_ranks
        self.walltime = walltime
        self.ranks_per_node = ranks_per_node
        self.gpus_per_node = gpus_per_node
        self.hyperthreads = hyperthreads
        self.cpu_bind = cpu_bind
        self.total_ranks = total_ranks
        self.total_gpus = total_gpus
        self.path = path
        self.script = script
        self.profile = profile


class JobSet:

    def __init__(self,
                 name,
                 test_run,
                 email,
                 mail_type,
                 project_allocation,
                 notes,
                 exawind_manager,
                 environment,
                 path):

        self.name = name
        self.email = email
        self.mail_type = mail_type
        self.project_allocation = project_allocation
        self.test_run = test_run
        self.notes = notes
        self.exawind_manager = exawind_manager
        self.environment = environment
        self.path = path

# ========================================================================
#
# Function for finding machine name
#
# ========================================================================


def find_machine_name():
    if os.getenv('LMOD_SYSTEM_NAME') == 'frontier':
        return 'frontier'
    if os.getenv('NREL_CLUSTER') == 'kestrel':
        if os.getenv('CRAY_CPU_TARGET') == 'x86-spr':
            return 'kestrel-cpu'
        elif os.getenv('CRAY_CPU_TARGET') == 'x86-genoa':
            return 'kestrel-gpu'
        else:
            print("Cannot determine host")
    else:
        print("Cannot determine host")
        exit(-1)


# ========================================================================
#
# Function for creating directory to contain this set of job results
#
# ========================================================================


def create_job_set_directory(job_set_name):
    now = datetime.datetime.now()
    path = '%s-%s' % (job_set_name, now.strftime("%Y-%m-%d-%H-%M"))
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the job set directory %s failed" % path)
    else:
        print("Successfully created the job set directory %s " % path)
    return path


# ========================================================================
#
# Function for creating directory to contain a single job's results
#
# ========================================================================


def create_job_directory(job_set_path, job_name):
    path = os.path.join(job_set_path, job_name)
    try:
        os.mkdir(path)
    except OSError:
        print("   Creation of the job directory %s failed" % path)
    else:
        print("   Directory: %s " % path)
    return path


# ========================================================================
#
# Function for checking file existence before job submission
#
# ========================================================================


def check_file_existence(job_executable, job_input_file, job_files_to_copy):
    #if os.path.isfile(job_executable) is False:
    #    print("   Executable does not exist: %s" % job_executable)
    #    exit(-1)
    if os.path.isfile(job_input_file) is False:
        print("   Input file does not exist: %s" % job_input_file)
        exit(-1)

    for myfile in job_files_to_copy:
        if os.path.isfile(myfile) is False:
            print("   Necessary file does not exist: %s" % myfile)
            exit(-1)


# ========================================================================
#
# Function for writing MPICH rank file
#
# ========================================================================


def write_reorder_file(job):
    nodes = job.nodes
    amr_wind_ranks_per_node = job.gpus_per_node
    total_amr_wind_ranks = nodes * amr_wind_ranks_per_node
    nalu_wind_ranks_per_node = job.ranks_per_node - amr_wind_ranks_per_node
    total_nalu_wind_ranks = nodes * nalu_wind_ranks_per_node
    total_ranks = total_nalu_wind_ranks + total_amr_wind_ranks
    
    print(f"AMR-Wind rank total: {total_amr_wind_ranks}")
    print(f"Nalu-Wind rank total: {total_nalu_wind_ranks}")

    cpu_map = ''
    for node in range(nodes):
        for amr_wind_rank in range(amr_wind_ranks_per_node):
            offset_amr_wind_rank = amr_wind_rank + (node * amr_wind_ranks_per_node)
            cpu_map = cpu_map + str(offset_amr_wind_rank) + ','
        for nalu_wind_rank in range(nalu_wind_ranks_per_node):
            offset_nalu_wind_rank = nalu_wind_rank + total_amr_wind_ranks + (node * nalu_wind_ranks_per_node)
            cpu_map = cpu_map + str(offset_nalu_wind_rank) + ','
        cpu_map = cpu_map + ',,'

    #cpu_map = cpu_map[:-1]
    with open(os.path.join(job.path, 'exawind.rank_map'), "w") as f:
        f.write(cpu_map)


# ========================================================================
#
# Function for writing single job script
#
# ========================================================================


def write_job_script(machine, job, job_set):
    print("   Writing job script...")

    job.script = "#!/bin/bash -l\n\n"
    job.script += "# Notes: " + job_set.notes + "\n\n"

    if machine == 'frontier' or machine == 'kestrel-cpu' or machine == 'kestrel-gpu':
        job.script += "#SBATCH -J " + job.name + "\n"
        job.script += "#SBATCH -o " + "%x.o%j\n"
        job.script += "#SBATCH -A " + job_set.project_allocation + "\n"
        job.script += "#SBATCH -t " + str(job.walltime) + "\n"
        job.script += "#SBATCH -p " + str(job.queue) + "\n"
        job.script += "#SBATCH -N " + str(job.nodes) + "\n"
        job.script += "#SBATCH -S " + str(0) + "\n"

    job.script += r"""
set -e
cmd() {
  echo "+ $@"
  eval "$@"
}
"""
    if job.mapping == 'exawind-all-gpu':
        job.script += ("echo \"Running with " + str(job.ranks_per_node)
                       + " ranks per node and " + str(job.ranks_per_gpu)
                       + " ranks per GPU on " + str(job.nodes)
                       + " nodes for a total of " + str(job.total_ranks)
                       + " ranks and " + str(job.total_gpus)
                       + " total GPUs with " + str(job.awind_ranks)
                       + " AMR-Wind ranks and " + str(job.nwind_ranks)
                       + " Nalu-Wind ranks...\"\n\n")
    elif job.mapping == 'amrwind-all-gpu':
        job.script += ("echo \"Running with " + str(job.ranks_per_node)
                       + " ranks per node and " + str(job.ranks_per_gpu)
                       + " ranks per GPU on " + str(job.nodes)
                       + " nodes for a total of " + str(job.total_ranks)
                       + " ranks and " + str(job.total_gpus)
                       + " total GPUs with " + str(job.awind_ranks)
                       + " AMR-Wind ranks...\"\n\n")
    elif job.mapping == 'pele-1-rank-per-gpu':
        job.script += ("echo \"Running with " + str(job.ranks_per_node)
                       + " ranks per node and " + str(job.ranks_per_gpu)
                       + " ranks per GPU on " + str(job.nodes)
                       + " nodes for a total of " + str(job.total_ranks)
                       + " ranks...\"\n\n")
    elif job.mapping == '1-rank-per-core':
        job.script += ("echo \"Running with " + str(job.ranks_per_node)
                       + " ranks per node on " + str(job.nodes)
                       + " nodes for a total of " + str(job.total_ranks)
                       + " ranks...\"\n\n")

    if machine == 'kestrel-cpu':
        job.script += "cmd \"module load PrgEnv-intel\"\n"
    if machine == 'kestrel-gpu':
        job.script += "cmd \"module load PrgEnv-gnu\"\n"

    if machine == 'frontier' or machine == 'kestrel-cpu' or machine == 'kestrel-gpu':
        job.script += ("cmd \"module load cray-python\"\n")
        job.script += ("#cmd \"export FI_MR_CACHE_MONITOR=memhooks\"\n")
        job.script += ("#cmd \"export FI_CXI_RX_MATCH_MODE=software\"\n")
        job.script += ("#cmd \"export MPICH_SMP_SINGLE_COPY_MODE=NONE\"\n")
        job.script += ("#cmd \"export MPICH_GPU_SUPPORT_ENABLED=1\"\n")
        job.script += ("cmd \"" + "export EXAWIND_MANAGER=" + job_set.exawind_manager + "\"\n")
        job.script += ("cmd \"source ${EXAWIND_MANAGER}/start.sh && spack-start\"\n")
        job.script += ("cmd \"spack env activate " + job_set.environment + "\"\n")
        job.script += ("cmd \"spack load " + job.spec + "\"\n")

    if machine == 'frontier' or machine == 'kestrel-gpu':
        if job.mapping == 'exawind-all-gpu' or job.mapping == 'exawind-nalu-cpu':
            job.script += ("cmd \"srun -N" + str(job.nodes)
                           + " -n" + str(job.nwind_ranks))
        if job.mapping == 'exawind-nalu-cpu':
            write_reorder_file(job)
            job.script += ("cmd \"export MPICH_RANK_REORDER_METHOD=3\"\n")
            job.script += ("cmd \"export MPICH_RANK_REORDER_FILE=exawind.rank_map\"\n")

        job.script += ("cmd \"" + job.pre_args + "srun -N" + str(job.nodes)
                       + " -n" + str(job.total_ranks)
                       + " --gpus-per-node=" + str(job.gpus_per_node)
                       + " --gpu-bind=closest")
        if job.mapping == 'exawind-all-gpu' or job.mapping == 'exawind-nalu-cpu':
            job.script += (" exawind --awind "
                           + str(job.awind_ranks) + " --nwind "
                           + str(job.nwind_ranks) + " "
                           + str(os.path.basename(job.input_file)) + " "
                           + job.post_args + "\"\n")
        elif job.mapping == 'pele-1-rank-per-gpu' or job.mapping == '1-rank-per-core':
            job.script += (" " + job.executable + " "
                           + str(os.path.basename(job.input_file)) + " "
                           + job.post_args + "\"\n")
        elif job.mapping == 'amrwind-all-gpu':
            if job.profile == "rocprof":
                job.script+=(" rocprof --roctx-trace --hip-trace --stats -o output.csv")
            job.script += (" amr_wind "
                           + str(os.path.basename(job.input_file)) + " "
                           + job.post_args + "\"\n")
    if machine == 'kestrel-cpu':
        job.script += ("cmd \"" + job.pre_args + "srun -N" + str(job.nodes)
                       + " -n" + str(job.total_ranks)
                       + " --ntasks-per-node=52 --distribution=cyclic:cyclic --cpu_bind=cores "
                       + job.executable + " "
                       + str(os.path.basename(job.input_file)) + " "
                       + job.post_args + "\"\n")

    # Write job script to file
    job.script_file = os.path.join(job.path, job.name + '.sh')
    job_script_file_handle = open(job.script_file, 'w')
    job_script_file_handle.write(job.script)
    job_script_file_handle.close()

    # Make the job script executable
    st = os.stat(job.script_file)
    os.chmod(job.script_file, st.st_mode | stat.S_IEXEC)


# ========================================================================
#
# Function for copying files to job working directory
#
# ========================================================================


def copy_files(job_files_to_copy, job_path):
    for myfile in job_files_to_copy:
        print("   Copying file %s" % myfile)
        shutil.copy(myfile, job_path)


# ========================================================================
#
# Function for submitting the job to the machine scheduler
#
# ========================================================================


def submit_job_script(machine, job, job_set):
    # Save current working directory
    mycwd = os.getcwd()
    # print("   Changing to directory " + job.path)
    os.chdir(job.path)

    if machine == 'frontier' or machine == 'kestrel-cpu' or machine == 'kestrel-gpu':
        batch = 'sbatch '

    print("   Submitting job...")
    command = batch + os.path.basename(job.script_file)
    if job_set.test_run is False:
        try:
            output = subprocess.check_output(
                command, stderr=subprocess.STDOUT, shell=True
            )
            print("   ".encode('ascii') + batch.encode('ascii') + "output: ".encode('ascii') + output)
        except subprocess.CalledProcessError as err:
            #print("   ".encode('ascii') + batch.encode('ascii') + "error: ".encode('ascii') + output)
            print(err.output)
    else:
        print("   TEST RUN. Real run would use the command:")
        print("     " + command)

    # Switch back to previous working directory
    os.chdir(mycwd)


# ========================================================================
#
# Function for printing some job info before submitting
#
# ========================================================================


def print_job_info(job_number, job):
    print("%s: %s" % (job_number, job.name))
    print("   Executable: %s" % job.executable)
    print("   Spec: %s" % job.spec)
    print("   Input file: %s" % job.input_file)
    print("   Mesh: %s" % job.mesh)
    print("   Queue: %s" % job.queue)
    print("   Mapping: %s" % job.mapping)
    print("   Compiler: %s" % job.compiler)
    print("   Nodes: %s" % job.nodes)
    print("   Minutes: %s" % job.minutes)
    print("   AMR-Wind Ranks: %s" % job.awind_ranks)
    print("   Nalu-Wind Ranks: %s" % job.nwind_ranks)
    print("   Pre args: %s" % job.pre_args)
    print("   Post args: %s" % job.post_args)
    print("   Profile: %s" % job.profile)


# ========================================================================
#
# Function for printing job set info before submitting
#
# ========================================================================


def print_job_set_info(job_set):
    print("Name: %s" % job_set.name)
    print("Project allocation: %s" % job_set.project_allocation)
    print("Email: %s" % job_set.email)
    print("Notes: %s" % job_set.notes)
    print("Exawind-Manager: %s" % job_set.exawind_manager)
    print("Environment: %s" % job_set.environment)
    if job_set.test_run is True:
        print("Performing test job submission")


# ========================================================================
#
# Function for populating job parameters according to machine type
#
# ========================================================================


def calculate_job_parameters(machine, job):
    if machine == 'frontier':
        job.walltime = job.minutes
        # AMD CPU logic
        job.hyperthreads = 2
        job.gpus_per_node = 8
        if job.mapping == 'exawind-all-gpu' or job.mapping == 'pele-1-rank-per-gpu' or job.mapping == 'amrwind-all-gpu':
            job.ranks_per_node = 8
            job.ranks_per_gpu = 1
        elif job.mapping == 'exawind-nalu-cpu':
            job.ranks_per_node = 62
            job.ranks_per_gpu = 1
    elif machine == 'kestrel-gpu':
        job.walltime = job.minutes
        # AMD CPU logic
        job.gpus_per_node = 4
        if job.mapping == 'exawind-all-gpu' or job.mapping == 'pele-1-rank-per-gpu' or job.mapping == 'amrwind-all-gpu':
            job.ranks_per_node = 4
            job.ranks_per_gpu = 1
        elif job.mapping == 'exawind-nalu-cpu':
            job.ranks_per_node = 124
            job.ranks_per_gpu = 1
    elif machine == 'kestrel-cpu':
        job.walltime = job.minutes
        # Intel CPU logic
        job.gpus_per_node = 0
        if job.mapping == '1-rank-per-core':
            job.ranks_per_node = 52
            job.ranks_per_gpu = 0

    job.total_ranks = int(job.nodes * job.ranks_per_node)
    job.total_gpus = int(job.nodes * job.gpus_per_node)


# ========================================================================
#
# Function for creating single instance of a job class
#
# ========================================================================


def create_job(job_number, job_instance, job_set_instance):
    job = Job(
      job_set_instance['name'] + "-" + str(job_number),  # name
      job_instance['queue'],           # queue
      job_instance['mapping'],         # mapping
      job_instance['compiler'],        # compiler
      job_instance['executable'],      # executable
      job_instance['spec'],            # spec
      job_instance['input_file'],      # input file
      job_instance['mesh'],            # mesh file
      job_instance['files_to_copy'],   # files to copy
      job_instance['nodes'],           # number of nodes
      job_instance['minutes'],         # number of job minutes
      job_instance['pre_args'],        # arguments before mpirun
      job_instance['post_args'],       # arguments after application
      job_instance['awind_ranks'],     # number of amr-wind ranks
      job_instance['nwind_ranks'],     # number of nalu-wind ranks
      0,   # walltime
      0,   # ranks_per_node
      0,   # gpus_per_node
      0,   # hyperthreads
      "",  # cpu_bind
      0,   # total_ranks
      0,   # total_gpus
      "",  # path
      "",  # script
      job_instance['profile']) # profile

    if job.pre_args is None:
        job.pre_args = ''

    if job.post_args is None:
        job.post_args = ''

    # Add input file into files to copy
    job.files_to_copy.append(job.input_file)

    return job


# ========================================================================
#
# Function for creating single instance of a job_set class
#
# ========================================================================


def create_job_set(job_set_instance, job_set_path):
    job_set = JobSet(
      job_set_instance['name'],               # name
      False,                                  # test_run
      job_set_instance['email'],              # email
      job_set_instance['mail_type'],          # mail_type
      job_set_instance['project_allocation'], # project_allocation
      job_set_instance['notes'],              # notes
      job_set_instance['exawind_manager'],    # exawind-manager
      job_set_instance['environment'],        # exawind-manager environment
      job_set_path                            # path
    )

    return job_set


# ========================================================================
#
# Main code to load job list and loop over jobs
#
# ========================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Job Scholar: Job submission powered by best practices.'
    )
    parser.add_argument('--test', dest='test_run', action='store_true',
                        help='perform a test job submission')
    parser.add_argument('job_set_file', type=argparse.FileType('r'),
                        help='file containing list of jobs in YAML format')
    parser.set_defaults(test_run=False)
    args = parser.parse_args()

    # Find the machine name or exit if machine is unsupported
    machine = find_machine_name()
    print("Machine detected as %s" % machine)

    # Load the job list file
    master_job_set = yaml.safe_load(args.job_set_file)
    job_set_instance = master_job_set['job_set']
    job_list_instance = job_set_instance['job_list']

    # Create main directory for this set of jobs
    job_set_path = create_job_set_directory(job_set_instance['name'])

    # Create the instance of the job_set class
    job_set = create_job_set(job_set_instance, job_set_path)

    # Copy yaml file used to create jobs into job set directory
    print("Copying file %s" % args.job_set_file.name)
    shutil.copy(args.job_set_file.name, job_set.path)

    if args.test_run is True:
        job_set.test_run = True

    # Print out the information on this job set before submitting
    print_job_set_info(job_set)

    # Main loop over jobs in list
    print("Submitting jobs...")
    job_counter = 1
    for job_instance in job_list_instance:
        # Create the instance of the job class
        job = create_job(job_counter, job_instance, job_set_instance)
        # Calculate some job parameters based on machine
        calculate_job_parameters(machine, job)
        # Print out the information on this job before submitting
        print_job_info(job_counter, job)
        # Check that some necessary files exist in the right place
        # before submitting
        check_file_existence(job.executable, job.input_file, job.files_to_copy)
        # Create this job's working directory
        job.path = create_job_directory(job_set.path, job.name)
        # Copy necessary files for the job listed by the user to the
        # job's working directory
        copy_files(job.files_to_copy, job.path)
        # Write the job script to be submitted
        write_job_script(machine, job, job_set)
        # Submit the job script for this job
        submit_job_script(machine, job, job_set)

        job_counter += 1


if __name__ == "__main__":
    main()
