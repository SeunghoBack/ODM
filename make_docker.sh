host_source_dir="/home/seungho/ppa_code/odm/ODM"
container_source_dir="/code/"
host_data_dir='/home/seungho/odm_project/uds/yp_data/datasets'
container_data_dir='/datasets'

# docker run -it --rm \
#   -v ${host_source_dir}:${container_source_dir} \
#   -v ${host_data_dir}:${container_data_dir} \
#   my_odm_image2

docker run -it --rm --gpus all\
    -v ${host_data_dir}:${container_data_dir} \
    fixed_odm2 --project-path /datasets project --fast-orthophoto \
    --orthophoto-resolution 0.1 --rerun-all

# docker run -it --rm \
#    -v ${host_data_dir}:${container_data_dir} \
#    fixed_odm2 --project-path /datasets project --fast-orthophoto \
#    --orthophoto-resolution 0.3 --min-num-features 2000 --rerun-all

# docker run -ti --rm \
#     -v ${host_data_dir}:${container_data_dir} --gpus all \
#     fixed_odm_gpu --project-path /datasets project --fast-orthophoto \
#     --orthophoto-resolution 0.3 --rerun-all  --min-num-features 2000
