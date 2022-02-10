docker run --name stable_mbrl_experiment_$1 \
    -it \
    --rm \
    --gpus all \
    --mount type=bind,source="$(pwd)"/src,target=/root/src \
    --mount type=bind,source="$(pwd)"/figures,target=/root/figures \
    --mount type=bind,source="$(pwd)"/videos,target=/root/videos \
    --mount type=bind,source="$(pwd)"/models,target=/root/models \
    --mount type=bind,source="$(pwd)"/data,target=/root/data \
    --dns 8.8.8.8 \
    thomasw219/lambda-mujoco-stable-mbrl:latest
