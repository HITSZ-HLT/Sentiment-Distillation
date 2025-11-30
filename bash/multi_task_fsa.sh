while getopts ':c:b:v:z:' opt
do
    case $opt in
        c) CUDA_IDS="$OPTARG" ;;
        b) subname="$OPTARG" ;;
        v) model_version="$OPTARG" ;;
        z) z_value="$OPTARG" ;;
        ?)  
        exit 1;;
    esac
done


tasks=("absa/asqp_rest16" "absa/opener" "absa/atsa_rest16" "absa/acsa_rest16")



for task in "${tasks[@]}"; do


    bash/fsa.sh -c ${CUDA_IDS} -d $task -b ${subname} -z ${z_value} -v ${model_version} -s 42
done


 