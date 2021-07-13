#!/usr/bin/env bash

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -ex

function init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'

    REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
    source ~/.bashrc
    export PATH="/root/miniconda3/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/TensorRT-6.0.1.5/lib:$LD_LIBRARY_PATH"
}

function print_usage() {
    echo -e "\n${RED}Usage${NONE}:
    ${BOLD}$0${NONE} [OPTION]"

    echo -e "\n${RED}Options${NONE}:
    ${BLUE}test${NONE}: run all unit tests
    ${BLUE}check_style${NONE}: run code style check
    "
}

function abort(){
    echo "Your change doesn't follow PaddlePaddle's code style." 1>&2
    echo "Please use pre-commit to check what is wrong." 1>&2
    exit 1
}

function check_style() {
    trap 'abort' 0
    set -e

    export PATH=/usr/bin:$PATH
    pre-commit install
    clang-format --version

    if ! pre-commit run -a ; then
        git diff
        exit 1
    fi

    trap : 0
}

function run_test_with_gpu() {
    unset CUDA_VISIBLE_DEVICES
    export FLAGS_fraction_of_gpu_memory_to_use=0.05
    
    mkdir -p ${REPO_ROOT}/build
    cd ${REPO_ROOT}/build

    if [ $# -eq 1 ];then
        cmake ..
    else
        cmake .. -$2=ON
    fi
    cat <<EOF
    ========================================
    Running unit tests with GPU...
    ========================================
EOF
    ctest --output-on-failure -j20
    cd ${REPO_ROOT}
    rm -rf ${REPO_ROOT}/build
}

function run_test_with_cpu() {
    export CUDA_VISIBLE_DEVICES=""

    mkdir -p ${REPO_ROOT}/build
    cd ${REPO_ROOT}/build
    if [ $# -eq 1 ];then
        cmake ..
    else
        cmake .. -$2=ON
    fi
    cat <<EOF
    =====================================================
    Running unit tests with CPU in the environment: $1
    =====================================================
EOF
    if [ "$#" == 2 ] && [ "$2" == "DIS_TESTING_SERIALLY" ]
    then
        ctest --output-on-failure 
    else
        ctest --output-on-failure -j20
    fi
    cd ${REPO_ROOT}
    rm -rf ${REPO_ROOT}/build
}

function run_single_fluid_test() {
    mkdir -p ${REPO_ROOT}/build
    cd ${REPO_ROOT}/build
    cmake .. -$1=ON
    ctest --output-on-failure -j20
    cd ${REPO_ROOT}
    rm -rf ${REPO_ROOT}/build
}

function run_test_with_fluid() {
    # declare -a envs=("py27" "py36" "py37")
    declare -a envs=("py37")
    for env in "${envs[@]}";do
        cd /work
        source ~/.bashrc
        export PATH="/root/miniconda3/bin:$PATH"
        source activate $env
        python -m pip install --upgrade pip
        echo "========================================"
        echo "Running tests in $env with paddlepaddle 1.8.5 .."
        echo `which pip`
        echo "========================================"
        pip install .
        pip install -r .teamcity/requirements_fluid.txt

        echo "========================================"
        echo "Running fluid unit tests with CPU..."
        echo "========================================"
        export CUDA_VISIBLE_DEVICES=""
        run_single_fluid_test "DIS_TESTING_FLUID"

        # clean env
        export LC_ALL=C.UTF-8
        export LANG=C.UTF-8
        xparl stop
    done
}

function run_example_test {
    for exp in QuickStart DQN
    do
        cp parl/tests/gym.py examples/${exp}/
        python examples/${exp}/train.py
    done
}

function run_import_test {
    export CUDA_VISIBLE_DEVICES=""

    mkdir -p ${REPO_ROOT}/build
    cd ${REPO_ROOT}/build

    cmake .. -DIS_TESTING_IMPORT=ON

    cat <<EOF
    ========================================
    Running import test...
    ========================================
EOF
    ctest --output-on-failure
    cd ${REPO_ROOT}
    rm -rf ${REPO_ROOT}/build
}

function run_docs_test {
    #export CUDA_VISIBLE_DEVICES=""

    mkdir -p ${REPO_ROOT}/build
    cd ${REPO_ROOT}/build

    cmake .. -DIS_TESTING_DOCS=ON 

    cat <<EOF
    ========================================
    Running docs test...
    ========================================
EOF
    ctest --output-on-failure
    cd ${REPO_ROOT}
    rm -rf ${REPO_ROOT}/build
}

function main() {
    set -e
    local CMD=$1
    
    init
    case $CMD in
        check_style)
            check_style
            ;;
        test)
            # test code compability in environments with various python versions
            #declare -a envs=("py36_torch" "py37_torch" "py27" "py36" "py37")
            declare -a envs=("py36" "py37" "py38")
            for env in "${envs[@]}";do
                cd /work
                source ~/.bashrc
                export PATH="/root/miniconda3/bin:$PATH"
                source activate $env
                python -m pip install --upgrade pip
                echo ========================================
                echo Running tests in $env ..
                echo `which pip`
                echo ========================================
                pip install .
                if [ \( $env == "py36" -o $env == "py37" -o $env == "py38" \) ]
                then
                    run_import_test # import parl test

                    pip install -r .teamcity/requirements.txt
                    pip install paddlepaddle==2.1.0
                    run_test_with_cpu $env
                    # uninstall paddlepaddle when testing remote module
                    pip uninstall -y paddlepaddle
                    run_test_with_cpu $env "DIS_TESTING_SERIALLY"
                    run_test_with_cpu $env "DIS_TESTING_REMOTE"
                else
                    echo ========================================
                    echo "in torch environment"
                    echo ========================================
                    pip install -r .teamcity/requirements_torch.txt
                    run_test_with_cpu $env "DIS_TESTING_TORCH"
                fi
                # clean env
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8
                xparl stop
            done

            pip install -r .teamcity/requirements.txt
            pip install /data/paddle_package/paddlepaddle_gpu-2.1.0.post101-cp38-cp38-linux_x86_64.whl
            run_test_with_gpu $env
            run_example_test $env

            run_test_with_fluid
            ############
            # run_docs_test

            ;;
        *)
            print_usage
            exit 0
            ;;
    esac
    echo "finished: ${CMD}"
}

main $@
