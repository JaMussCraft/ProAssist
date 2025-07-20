export OMP_NUM_THREADS=4

cd mmassist/datasets/generate
OMP_NUM_THREADS=4 python generate_ego4d_dialog.py --job_name gen_ego4d --num_nodes 32 --tasks_per_node 2 --timeout_min 360
OMP_NUM_THREADS=4 python generate_holoassist.py --job_name gen_hlass --num_nodes 32 --tasks_per_node 2 --timeout_min 360
OMP_NUM_THREADS=4 python generate_egoexolearn.py --job_name gen_egoexo --num_nodes 32 --tasks_per_node 2 --timeout_min 360
OMP_NUM_THREADS=4 python generate_epickitchens.py --job_name gen_ek --num_nodes 32 --tasks_per_node 2 --timeout_min 360
OMP_NUM_THREADS=4 python generate_wtag.py --job_name gen_wtag --num_nodes 1 --tasks_per_node 2 --timeout_min 360
OMP_NUM_THREADS=4 python generate_assembly101.py --job_name gen_assem --num_nodes 32 --tasks_per_node 2 --timeout_min 360