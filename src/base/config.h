#ifndef CONFIG_H
#define CONFIG_H

/*DEBUG*/
#define KERNEL_DEBUG

/* Print debug messages */
/* #undef KERNEL_PRINT_DBG_MSGS */

/*Use log */
#define USE_LOG

/*Use OpenMP */
#define USE_OPENMP


/*Use PyTorch */
#define USE_PYTORCH

/*Use FVM*/
/* #undef USE_FVM */

/*Use FEM*/
/* #undef USE_FEM */

/*Use Trilinos*/
/* #undef USE_TRILINOS */

/*Trilinos integer type*/
/* #undef USE_TRILINOS_LONG_LONG_TYPE */

/*Path to the datasets*/
#define DATA_SET_FOLDER "/home/alex/qi3/ce_rl/data"

/*Path to the datasets for testing*/
/* #undef TEST_DATA_SET_DIRECTORY */

/* Path to the project*/
/* #undef PROJECT_PATH */

/*Use OpenCV */
/* #undef USE_OPEN_CV */

/*Use discretization module*/
/* #undef USE_DISCRETIZATION */

/*Use Numerics module*/
/* #undef USE_NUMERICS */

/*Use Rigid body dynamics module*/
#define USE_RIGID_BODY_DYNAMICS




#endif

