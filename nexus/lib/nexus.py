##################################################################
##  (c) Copyright 2015-  by Jaron T. Krogel                     ##
##################################################################


#====================================================================#
#  nexus.py                                                          #
#    Gathering place for user-facing Nexus functions.  Management    #
#    of user-provided settings.                                      #
#                                                                    #
#  Content summary:                                                  #
#    Settings                                                        #
#      Class to set 'global' Nexus data.                             #
#                                                                    #
#    settings                                                        #
#      A single Settings instance users interact with as a function. #
#                                                                    #
#    run_project                                                     #
#      User interface to the ProjectManager.                         #
#      Runs all simulations generated by the user.                   #
#                                                                    #
#====================================================================#


import os

from versions import nexus_version,current_versions,policy_versions,check_versions

from generic import obj
from developer import error,log

from nexus_base      import NexusCore,nexus_core,nexus_noncore,nexus_core_noncore,restore_nexus_core_defaults,nexus_core_defaults
from machines        import Job,job,Machine,Supercomputer,get_machine
from simulation      import generate_simulation,input_template,multi_input_template,generate_template_input,generate_multi_template_input,graph_sims
from project_manager import ProjectManager

from structure       import Structure,generate_structure,generate_cell,read_structure
from physical_system import PhysicalSystem,generate_physical_system
from pseudopotential import Pseudopotential,Pseudopotentials,ppset
from basisset        import BasisSets
from bundle          import bundle

from pwscf   import Pwscf  , PwscfInput  , PwscfAnalyzer  , generate_pwscf_input  , generate_pwscf
from gamess  import Gamess , GamessInput , GamessAnalyzer , generate_gamess_input , generate_gamess, FormattedGroup
from vasp    import Vasp   , VaspInput   , VaspAnalyzer   , generate_vasp_input   , generate_vasp
from qmcpack import Qmcpack, QmcpackInput, QmcpackAnalyzer, generate_qmcpack_input, generate_qmcpack
from quantum_package import QuantumPackage,QuantumPackageInput,QuantumPackageAnalyzer,generate_quantum_package_input,generate_quantum_package
from pyscf_sim import Pyscf, PyscfInput, PyscfAnalyzer, generate_pyscf_input, generate_pyscf

from qmcpack_converters import Pw2qmcpack , Pw2qmcpackInput , Pw2qmcpackAnalyzer , generate_pw2qmcpack_input , generate_pw2qmcpack
from qmcpack_converters import Convert4qmc, Convert4qmcInput, Convert4qmcAnalyzer, generate_convert4qmc_input, generate_convert4qmc
from qmcpack_converters import PyscfToAfqmc, PyscfToAfqmcInput, PyscfToAfqmcAnalyzer, generate_pyscf_to_afqmc_input, generate_pyscf_to_afqmc

from pwscf_postprocessors import PP      , PPInput      , PPAnalyzer      , generate_pp_input      , generate_pp
from pwscf_postprocessors import Dos     , DosInput     , DosAnalyzer     , generate_dos_input     , generate_dos
from pwscf_postprocessors import Bands   , BandsInput   , BandsAnalyzer   , generate_bands_input   , generate_bands
from pwscf_postprocessors import Projwfc , ProjwfcInput , ProjwfcAnalyzer , generate_projwfc_input , generate_projwfc
from pwscf_postprocessors import Cppp    , CpppInput    , CpppAnalyzer    , generate_cppp_input    , generate_cppp
from pwscf_postprocessors import Pwexport, PwexportInput, PwexportAnalyzer, generate_pwexport_input, generate_pwexport

from qmcpack import loop,linear,cslinear,vmc,dmc
from qmcpack import generate_jastrows,generate_jastrow,generate_jastrow1,generate_jastrow2,generate_jastrow3,generate_opt,generate_opts
from qmcpack import generate_cusp_correction

from debug import *


#set the machine if known, otherwise user will provide
hostmachine = Machine.get_hostname()
if Machine.exists(hostmachine):
    Job.machine = hostmachine
    ProjectManager.machine = Machine.get(hostmachine)
#end if



class Settings(NexusCore):
    singleton = None

    machine_vars = set('''
        machine         account         machine_info    interactive_cores
        machine_mode    user
        '''.split())

    core_assign_vars = set('''
        status_only     generate_only   runs            results 
        pseudo_dir      sleep           local_directory remote_directory 
        monitor         skip_submit     load_images     stages          
        verbose         debug           trace           progress_tty
        graph_sims      command_line
        '''.split())

    core_process_vars = set('''
        file_locations  mode  status
        '''.split())

    noncore_assign_vars = set('''
        basis_dir
        '''.split())

    noncore_process_vars = set()
    
    gamess_vars  = set('''
        ericfmt         mcppath
        '''.split())
    
    pwscf_vars   = set('''
        vdw_table
        '''.split())

    qm_package_vars = set('''
        qprc
        '''.split())

    nexus_core_vars    = core_assign_vars    | core_process_vars
    nexus_noncore_vars = noncore_assign_vars | noncore_process_vars
    nexus_vars         = nexus_core_vars     | nexus_noncore_vars
    allowed_vars       = nexus_vars | machine_vars \
                       | gamess_vars | pwscf_vars | qm_package_vars


    @staticmethod
    def kw_set(vars,source=None):
        kw = obj()
        if source!=None:
            for n in vars:
                if n in source:
                    kw[n]=source[n]
                    del source[n]
                #end if
            #end for
        #end if
        return kw
    #end def null_kw_set


    def __init__(self):
        if Settings.singleton is None:
            Settings.singleton = self
        else:
            self.error('attempted to create a second Settings object\nplease just use the original')
        #end if
    #end def __init__


    def error(self,message,header='settings',exit=True,trace=True):
        NexusCore.error(self,message,header,exit,trace)
    #end def error


    # sets up Nexus core class behavior and passes information to broader class structure
    def __call__(self,**kwargs):
        kwargs = obj(**kwargs)

        # guard against invalid settings
        not_allowed = set(kwargs.keys()) - Settings.allowed_vars
        if len(not_allowed)>0:
            self.error('unrecognized variables provided\nyou provided: {0}\nallowed variables are: {1}'.format(sorted(not_allowed),sorted(Settings.allowed_vars)))
        #end if

        # restore default core default settings
        restore_nexus_core_defaults()

        # process command line inputs, if any
        if 'command_line' in kwargs:
            nexus_core.command_line = kwargs.command_line
        #end if
        if nexus_core.command_line:
            self.process_command_line_settings(kwargs)
        #end if

        NexusCore.write_splash()

        # print version information
        try:
            from versions import versions
            if versions is not None:
                err,s,serr = versions.check(write=False,full=True)
                self.log(s)
            #end if
        except Exception:
            None
        #end try

        self.log('Applying user settings')

        # assign simple variables
        for name in Settings.core_assign_vars:
            if name in kwargs:
                nexus_core[name] = kwargs[name]
            #end if
        #end for

        # assign simple variables
        for name in Settings.noncore_assign_vars:
            if name in kwargs:
                nexus_noncore[name] = kwargs[name]
            #end if
        #end for

        # extract settings based on keyword groups
        kw        = Settings.kw_set(Settings.nexus_vars     ,kwargs)
        mach_kw   = Settings.kw_set(Settings.machine_vars   ,kwargs)
        gamess_kw = Settings.kw_set(Settings.gamess_vars    ,kwargs)
        pwscf_kw  = Settings.kw_set(Settings.pwscf_vars     ,kwargs)
        qm_pkg_kw = Settings.kw_set(Settings.qm_package_vars,kwargs)
        if len(kwargs)>0:
            self.error('some settings keywords have not been accounted for\nleftover keywords: {0}\nthis is a developer error'.format(sorted(kwargs.keys())))
        #end if


        # copy input settings
        self.transfer_from(mach_kw.copy())
        self.transfer_from(gamess_kw.copy())
        self.transfer_from(pwscf_kw.copy())

        # process machine settings
        self.process_machine_settings(mach_kw)

        # process nexus core settings
        self.process_core_settings(kw)

        # process nexus noncore settings
        self.process_noncore_settings(kw)

        # transfer select core data to the global namespace
        nexus_core_noncore.transfer_from(nexus_core,list(nexus_core_noncore.keys()))
        nexus_noncore.set(**nexus_core_noncore.copy()) # prevent write to core namespace

        # copy final core and noncore settings
        self.transfer_from(nexus_core.copy())
        self.transfer_from(nexus_noncore.copy())


        # process gamess settings
        Gamess.restore_default_settings()
        Gamess.settings(**gamess_kw)

        # process pwscf settings
        Pwscf.restore_default_settings()
        Pwscf.settings(**pwscf_kw)

        # process quantum package settings
        QuantumPackage.restore_default_settings()
        QuantumPackage.settings(**qm_pkg_kw)

        return
    #end def __call__


    def process_command_line_settings(self,script_settings):
        from optparse import OptionParser
        usage = '''usage: %prog [options]'''
        version = '{}.{}.{}'.format(*nexus_version)
        parser = OptionParser(usage=usage,add_help_option=True,version='%prog '+version)

        parser.add_option('--status_only',dest='status_only',
                          action='store_true',default=False,
                          help='Report status of all simulations and then exit.'
                          )
        parser.add_option('--status',dest='status',
                          default='none',
                          help="Controls displayed simulation status information.  May be set to one of 'standard', 'active', 'failed', or 'ready'."
                          )
        parser.add_option('--generate_only',dest='generate_only',
                          action='store_true',default=False,
                          help='Write inputs to all simulations and then exit.  Note that no dependencies are processed, e.g. if one simulation depends on another for an orbital file location or for a relaxed structure, this information will not be present in the generated input file for that simulation since no simulations are actually run with this option.'
                          )
        parser.add_option('--graph_sims',dest='graph_sims',
                          action='store_true',default=False,
                          help='Display a graph of simulation workflows, then exit.'
                          )
        parser.add_option('--progress_tty',dest='progress_tty',
                          action='store_true',default=False,
                          help='Print abbreviated polling messages.'
                          )
        parser.add_option('--sleep',dest='sleep',
                          default='none',
                          help='Number of seconds between polls.  At each poll, simulations are actually run provided all simulations they depend on have successfully completed (default={0}).'.format(nexus_core_defaults.sleep)
                          )
        parser.add_option('--machine',dest='machine',
                          default='none',
                          help="(Required) Name of the machine the simulations will be run on.  Workstations with between 1 and 128 cores may be specified by 'ws1' to 'ws128' (works for any machine where only mpirun is used).  For a complete listing of currently available machines (including those at HPC centers) please see the manual."
                          )
        parser.add_option('--account',dest='account',
                          default='none',
                          help='Account name required to submit jobs at some HPC centers.'
                          )
        parser.add_option('--runs',dest='runs',
                          default='none',
                          help='Directory to perform all runs in.  Simulation paths are appended to this directory (default={0}).'.format(nexus_core_defaults.runs)
                          )
        parser.add_option('--results',dest='results',
                          default='none',
                          help="Directory to copy out lightweight results data.  If set to '', results will not be stored outside of the runs directory (default={0}).".format(nexus_core_defaults.results)
                          )
        parser.add_option('--local_directory',dest='local_directory',
                          default='none',
                          help='Base path where runs and results directories will be created (default={0}).'.format(nexus_core_defaults.local_directory)
                          )
        parser.add_option('--pseudo_dir',dest='pseudo_dir',
                          default='none',
                          help='Path to directory containing pseudopotential files (required if running with pseudopotentials).'
                          )
        parser.add_option('--basis_dir',dest='basis_dir',
                          default='none',
                          help='Path to directory containing basis set files (useful if running gaussian based QMC workflows).'
                          )
        parser.add_option('--ericfmt',dest='ericfmt',
                          default='none',
                          help='Path to the ericfmt file used with GAMESS (required if running GAMESS).'
                          )
        parser.add_option('--mcppath',dest='mcppath',
                          default='none',
                          help='Path to the mcpdata file used with GAMESS (optional for most workflows)'
                          )
        parser.add_option('--vdw_table',dest='vdw_table',
                          default='none',
                          help='Path to the vdw_table file used with Quantum Espresso (required only if running Quantum Espresso with van der Waals functionals).'
                          )
        parser.add_option('--qprc',dest='qprc',
                          default='none',
                          help='Path to the quantum_package.rc file used with Quantum Package.'
                          )

        # parse the command line inputs
        options,files_in = parser.parse_args()
        opt = obj()
        opt.transfer_from(options.__dict__)

        # check that all options are allowed (developer check)
        invalid = set(opt.keys())-Settings.allowed_vars
        if len(invalid)>0:
            self.error('invalid command line settings encountered\ninvalid settings: {0}\nthis is a developer error'.format(sorted(invalid)))
        #end if

        # pre-process options, full processing occurs upon return
        boolean_options = set(['status_only','generate_only','progress_tty'])
        real_options = set(['sleep'])
        for ropt in real_options:
            if opt[ropt]!='none':
                try:
                    opt[ropt] = float(opt[ropt])
                except:
                    self.error("command line option '{0}' must be a real value\nyou provided: {1}\nplease try again".format(ropt,opt[ropt]))
                #end try
            #end if
        #end for

        # override script settings with command line settings
        for name,value in opt.items():
            bool_name = name in boolean_options
            if (bool_name and value) or (not bool_name and value!='none'):
                script_settings[name] = value
            #end if
        #end for

    #end def process_command_line_settings


    def process_machine_settings(self,mset):
        Job.restore_default_settings()
        ProjectManager.restore_default_settings()
        mid_set = set()
        if 'machine_info' in mset:
            machine_info = mset.machine_info
            if isinstance(machine_info,dict) or isinstance(machine_info,obj):
                for machine_name,minfo in machine_info.items():
                    mname = machine_name.lower()
                    if Machine.exists(mname):
                        machine = Machine.get(mname)
                        machine.restore_default_settings()
                        machine.incorporate_user_info(minfo)
                        mid_set.add(id(machine))
                    else:
                        self.error('machine {0} is unknown\n  cannot set machine_info'.format(machine_name))
                    #end if
                #end for
            else:
                self.error('machine_info must be a dict or obj\n  you provided type '+machine_info.__class__.__name__)
            #end if
        #end if
        if 'machine' in mset:
            machine_name = mset.machine
            if not Machine.exists(machine_name):
                self.error('machine {0} is unknown'.format(machine_name))
            #end if
            Job.machine = machine_name
            machine = Machine.get(machine_name)
            ProjectManager.machine = machine
            if machine is not None and id(machine) not in mid_set:
                machine.restore_default_settings()
            #end if
            if 'account' in mset:
                account = mset.account
                if not isinstance(account,str):
                    self.error('account for {0} must be a string\nyou provided: {1}'.format(machine_name,account))
                #end if
                ProjectManager.machine.account = account
            #end if
            if 'user' in mset:
                user = mset.user
                if not isinstance(user,str):
                    self.error('user for {0} must be a string\nyou provided: {1}'.format(machine_name,user))
                #end if
                ProjectManager.machine.user = user
            #end if
            if 'machine_mode' in mset:
                machine_mode = mset.machine_mode
                if machine_mode in Machine.modes:
                    machine_mode = Machine.modes[machine_mode]
                #end if
                if machine_mode==Machine.modes.interactive:
                    if ProjectManager.machine==None:
                        ProjectManager.class_error('no machine specified for interactive mode')
                    #end if
                    if not isinstance(ProjectManager.machine,Supercomputer):
                        self.error('interactive mode is not supported for machine type '+ProjectManager.machine.__class__.__name__)
                    #end if
                    if not 'interactive_cores' in mset:
                        self.error('interactive mode requested, but interactive_cores not set')
                    #end if
                    ProjectManager.machine = ProjectManager.machine.interactive_representation(mset.interactive_cores)
                    Job.machine = ProjectManager.machine.name
                #end if
            #end if
        #end if
    #end def process_machine_settings


    def process_core_settings(self,kw):
        # process project manager settings
        if nexus_core.debug:
            nexus_core.verbose = True
        #end if
        if 'status' in kw:
            if kw.status==None or kw.status==False:
                nexus_core.status = nexus_core.status_modes.none
            elif kw.status==True:
                nexus_core.status = nexus_core.status_modes.standard
            elif kw.status in nexus_core.status_modes:
                nexus_core.status = nexus_core.status_modes[kw.status]
            else:
                self.error('invalid status mode specified: {0}\nvalid status modes are: {1}'.format(kw.status,sorted(nexus_core.status_modes.keys())))
            #end if
        #end if
        if nexus_core.status_only and nexus_core.status==nexus_core.status_modes.none:
            nexus_core.status = nexus_core.status_modes.standard
        #end if
        if 'mode' in kw:
            if kw.mode in nexus_core.modes:
                nexus_core.mode = kw.mode
            else:
                self.error('invalid mode specified: {0}\nvalid modes are: {1}'.format(kw.mode,sorted(nexus_core.modes.keys())))
            #end if
        #end if
        mode  = nexus_core.mode
        modes = nexus_core.modes
        if mode==modes.stages:
            stages = nexus_core.stages
        elif mode==modes.all:
            stages = list(nexus_core.primary_modes)
        else:
            stages = [kw.mode]
        #end if
        allowed_stages = set(nexus_core.primary_modes)
        if isinstance(stages,str):
            stages = [stages]
        #end if
        if len(stages)==0:
            stages = list(nexus_core.primary_modes)
        elif 'all' in stages:
            stages = list(nexus_core.primary_modes)
        else:
            forbidden = set(nexus_core.stages)-allowed_stages
            if len(forbidden)>0:
                self.error('some stages provided are not primary stages.\n  You provided '+str(list(forbidden))+'\n  Options are '+str(list(allowed_stages)))
            #end if
        #end if
        # overide user input and always use stages mode 
        # keep processing code above in case a change is desired in the future
        nexus_core.mode       = modes.stages
        nexus_core.stages     = stages
        nexus_core.stages_set = set(nexus_core.stages)

        # process simulation settings
        if 'local_directory' in kw:
            nexus_core.file_locations.append(kw.local_directory)
        #end if
        if 'file_locations' in kw:
            fl = kw.file_locations
            if isinstance(fl,str):
                nexus_core.file_locations.extend([fl])
            else:
                nexus_core.file_locations.extend(list(fl))
            #end if
        #end if
        if not 'pseudo_dir' in kw:
            nexus_core.pseudopotentials = Pseudopotentials()
        else:
            pseudo_dir = kw.pseudo_dir
            nexus_core.file_locations.append(pseudo_dir)
            if not os.path.exists(pseudo_dir):
                self.error('pseudo_dir "{0}" does not exist'.format(pseudo_dir),trace=False)
            #end if
            files = os.listdir(pseudo_dir)
            ppfiles = []
            for f in files:
                pf = os.path.join(pseudo_dir,f)
                if os.path.isfile(pf):
                    ppfiles.append(pf)
                #end if
            #end for
            nexus_core.pseudopotentials = Pseudopotentials(ppfiles)        
        #end if
    #end def process_core_settings


    def process_noncore_settings(self,kw):
        if not 'basis_dir' in kw:
            nexus_noncore.basissets = BasisSets()
        else:
            basis_dir = kw.basis_dir
            nexus_core.file_locations.append(basis_dir)
            if not os.path.exists(basis_dir):
                self.error('basis_dir "{0}" does not exist'.format(basis_dir),trace=False)
            #end if
            files = os.listdir(basis_dir)
            bsfiles = []
            for f in files:
                pf = os.path.join(basis_dir,f)
                if os.path.isfile(pf):
                    bsfiles.append(pf)
                #end if
            #end for
            nexus_noncore.basissets = BasisSets(bsfiles)        
        #end if
    #end def process_noncore_settings
#end class Settings


# create settings functor for UI
settings = Settings()


# test needed
def run_project(*args,**kwargs):
    if nexus_core.graph_sims:
        graph_sims()
    #end if
    pm = ProjectManager()
    pm.add_simulations(*args,**kwargs)
    pm.run_project()
    return pm
#end def run_project







# test needed
# read input function
#   place here for now as it depends on all other input functions
def read_input(filepath,format=None):
    if not os.path.exists(filepath):
        error('cannot read input file\nfile does not exist: {0}'.format(filepath),'read_input')
    #end if
    if format is None:
        if filepath.endswith('in.xml'):
            format = 'qmcpack'
        else:
            error('cannot identify file format\nplease provide format for file: {0}'.format(filepath))
        #end if
    #end if
    format = format.lower()
    if format=='qmcpack':
        input = QmcpackInput(filepath)
    elif format=='pwscf':
        input = PwscfInput(filepath)
    elif format=='gamess':
        input = GamessInput(filepath)
    else:
        error('cannot read input file\nfile format "{0}" is unsupported'.format(format))
    #end if
    return input
#end def read_input
