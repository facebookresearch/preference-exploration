#! /usr/bin/zsh

if [ "$#" -ne 4 ]
then
  echo "Incorrect number of arguments $#"
  exit 1
fi

while getopts g:b:e:c: flag
do
    case "${flag}" in
        g) gpu=${OPTARG};;
        b) begin_init_seed=${OPTARG};;
        e) end_init_seed=${OPTARG};;
        c) cnt=${OPTARG};;
    esac
done

echo "Using $(which python)"

problem_strs=(
    "vehiclesafety_5d3d_kumaraswamyproduct"
    "dtlz2_8d4d_negl1dist"
    "osy_6d8d_piecewiselinear"
    "carcabdesign_7d9d_piecewiselinear"

    "vehiclesafety_5d3d_piecewiselinear"
    "dtlz2_8d4d_piecewiselinear"
    "osy_6d8d_sigmodconstraints"
    "carcabdesign_7d9d_linear"
)

for i in {$begin_init_seed..$end_init_seed}
do
    echo "Running the simulation round $i"

    shuffled=( $(shuf -e "${problem_strs[@]}") )

    for problem_str in ${shuffled}; do
        # echo "GPU=$gpu runs=$num_sim $problem_str"
        c="CUDA_VISIBLE_DEVICES=$gpu nice -n 19 python within_session_sim.py --problem_str=$problem_str --noisy=False --init_seed=$i --gen_method=qnei --kernel=default --comp_noise_type=$cnt --device=$gpu"
        echo $c
        eval $c
    done
done

